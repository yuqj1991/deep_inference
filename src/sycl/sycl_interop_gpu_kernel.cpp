/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <CL/sycl.hpp>

#include "common/utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/zero_pad_struct.h"
#include "sycl/level_zero_utils.hpp"
#include "sycl/sycl_c_types_map.hpp"
#include "sycl/sycl_interop_gpu_kernel.hpp"
#include "sycl/sycl_stream.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

static void set_scalar_arg(
        cl::sycl::handler &cgh, int index, size_t size, const void *value) {
    switch (size) {
        case sizeof(uint8_t):
            cgh.set_arg(index, *static_cast<const uint8_t *>(value));
            break;
        case sizeof(uint16_t):
            cgh.set_arg(index, *static_cast<const uint16_t *>(value));
            break;
        case sizeof(uint32_t):
            cgh.set_arg(index, *static_cast<const uint32_t *>(value));
            break;
        case sizeof(uint64_t):
            cgh.set_arg(index, *static_cast<const uint64_t *>(value));
            break;
        case sizeof(zero_pad_mask_t):
            cgh.set_arg(index, *static_cast<const zero_pad_mask_t *>(value));
            break;
        default:
            assert(!"Please add another case");
            throw std::runtime_error("Internal error");
    }
}

static status_t create_ocl_kernel(
        gpu::ocl::ocl_wrapper_t<cl_kernel> &ocl_kernel, cl_device_id dev,
        cl_context ctx, const std::vector<unsigned char> &binary,
        const std::string &kernel_name) {
    cl_int err;
    const unsigned char *binary_buffer = binary.data();
    size_t binary_size = binary.size();
    assert(binary_size > 0);

    auto program = gpu::ocl::make_ocl_wrapper(clCreateProgramWithBinary(
            ctx, 1, &dev, &binary_size, &binary_buffer, nullptr, &err));
    OCL_CHECK(err);
    err = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
    OCL_CHECK(err);

    ocl_kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    OCL_CHECK(err);

    return status::success;
}

static status_t get_kernel_arg_types(
        std::vector<gpu::compute::scalar_type_t> &arg_types,
        cl_kernel ocl_kernel) {
    cl_uint nargs;
    OCL_CHECK(clGetKernelInfo(
            ocl_kernel, CL_KERNEL_NUM_ARGS, sizeof(nargs), &nargs, nullptr));

    arg_types.resize(nargs);

    for (int i = 0; i < nargs; i++) {
        gpu::compute::scalar_type_t type;
        CHECK(gpu::ocl::get_ocl_kernel_arg_type(
                &type, ocl_kernel, i, /*allow_undef=*/true));
        arg_types[i] = type;
    }

    return status::success;
}

status_t sycl_interop_gpu_kernel_t::realize(
        gpu::compute::kernel_t *kernel, const engine_t *engine) const {
    assert(state_ == state_t::binary);
    if (binary_.empty()) return status::success;
    auto *sycl_engine = utils::downcast<const sycl_gpu_engine_t *>(engine);

    std::unique_ptr<cl::sycl::kernel> sycl_kernel;
    std::vector<gpu::compute::scalar_type_t> arg_types;

    if (sycl_engine->backend() == backend_t::opencl) {
        gpu::ocl::ocl_wrapper_t<cl_kernel> ocl_kernel;
        CHECK(create_ocl_kernel(ocl_kernel, sycl_engine->ocl_device(),
                sycl_engine->ocl_context(), binary_, binary_name_));
        CHECK(get_kernel_arg_types(arg_types, ocl_kernel));

        cl_program ocl_program;
        OCL_CHECK(clGetKernelInfo(ocl_kernel, CL_KERNEL_PROGRAM,
                sizeof(ocl_program), &ocl_program, nullptr));

        cl::sycl::program sycl_program(sycl_engine->context(), ocl_program);

        sycl_kernel.reset(
                new cl::sycl::kernel(sycl_program.get_kernel(binary_name_)));
    } else if (sycl_engine->backend() == backend_t::level0) {
#ifdef DNNL_WITH_LEVEL_ZERO
        // FIXME: This does not work for multi-GPU systems. OpenCL engine
        // should be created based on the L0 device to ensure that the program
        // is created for the same physical device that was used to create the
        // binary. However, OpenCL does not provide any API to match its
        // devices with L0.
        //
        // Currently we always create an OpenCL engine for the 0th device at
        // binary creation time and here.
        gpu::ocl::ocl_engine_factory_t f(engine_kind::gpu);
        engine_t *ocl_engine_ptr;
        CHECK(f.engine_create(&ocl_engine_ptr, 0));
        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t> ocl_engine;
        ocl_engine.reset(
                utils::downcast<gpu::ocl::ocl_gpu_engine_t *>(ocl_engine_ptr));

        gpu::ocl::ocl_wrapper_t<cl_kernel> ocl_kernel;
        CHECK(create_ocl_kernel(ocl_kernel, ocl_engine->device(),
                ocl_engine->context(), binary_, binary_name_));
        CHECK(get_kernel_arg_types(arg_types, ocl_kernel));

        CHECK(sycl_create_kernel_with_level_zero(
                sycl_kernel, sycl_engine, binary_, binary_name_));
#else
        assert(!"not expected");
        return status::invalid_arguments;
#endif
    } else {
        assert(!"not expected");
        return status::invalid_arguments;
    }

    (*kernel) = gpu::compute::kernel_t(
            new sycl_interop_gpu_kernel_t(*sycl_kernel, arg_types));

    return status::success;
}

status_t sycl_interop_gpu_kernel_t::parallel_for(stream_t &stream,
        const gpu::compute::nd_range_t &range,
        const gpu::compute::kernel_arg_list_t &arg_list) const {
    assert(state_ == state_t::kernel);

    if (range.is_zero()) return status::success;
    auto *sycl_stream = utils::downcast<sycl::sycl_stream_t *>(&stream);
    auto &queue = sycl_stream->queue();

    auto event = queue.submit([&](cl::sycl::handler &cgh) {
#ifdef DNNL_SYCL_DPCPP
        cgh.depends_on(sycl_stream->get_deps());
#endif
        for (int i = 0; i < arg_list.nargs(); ++i) {
            auto &arg = arg_list.get(i);
            if (arg.is_global()) {
                auto *mem_storage
                        = static_cast<const memory_storage_t *>(arg.value());
                if (*mem_storage) {
                    auto *sycl_mem_storage = utils::downcast<
                            const sycl_memory_storage_base_t *>(mem_storage);
                    switch (sycl_mem_storage->memory_kind()) {
                        case memory_kind::buffer: {
                            auto *m = utils::downcast<
                                    const sycl_buffer_memory_storage_t *>(
                                    mem_storage);
                            auto &sycl_buf = m->buffer();
                            cgh.set_arg((int)i,
                                    sycl_buf.get_access<
                                            cl::sycl::access::mode::read_write>(
                                            cgh));
                            break;
                        }
#ifdef DNNL_SYCL_DPCPP
                        case memory_kind::usm: {
                            auto *m = utils::downcast<
                                    const sycl_usm_memory_storage_t *>(
                                    mem_storage);
                            cgh.set_arg((int)i, m->usm_ptr());
                            break;
                        }
#endif
                        default: assert(!"not expected");
                    }
                } else {
                    cgh.set_arg((int)i, nullptr);
                }
            } else if (arg.is_local()) {
                auto acc = cl::sycl::accessor<uint8_t, 1,
                        cl::sycl::access::mode::read_write,
                        cl::sycl::access::target::local>(
                        cl::sycl::range<1>(arg.size()), cgh);
                cgh.set_arg((int)i, acc);
            } else {
                if (arg_types_[i] == gpu::compute::scalar_type_t::undef) {
                    assert(!"not expected");
                }
                typename std::aligned_storage<sizeof(float),
                        sizeof(float)>::type tmp_storage;
                void *cast_storage = &tmp_storage;
                auto cvt_arg = gpu::compute::kernel_arg_t::cast(
                        arg_types_[i], arg, cast_storage);
                set_scalar_arg(cgh, (int)i, cvt_arg.size(), cvt_arg.value());
            }
        }
        if (range.local_range()) {
            auto sycl_nd_range = to_sycl_nd_range(range);
            cgh.parallel_for(sycl_nd_range, *sycl_kernel_);
        } else {
            auto sycl_range = to_sycl_range(range);
            cgh.parallel_for(sycl_range, *sycl_kernel_);
        }
    });

    sycl_stream->set_deps({event});
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
