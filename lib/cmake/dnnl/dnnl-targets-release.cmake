#----------------------------------------------------------------
# Generated CMake target import file for configuration "release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "DNNL::dnnl" for configuration "release"
set_property(TARGET DNNL::dnnl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(DNNL::dnnl PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "-lpthread;dl"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libdnnl.so.1.7"
  IMPORTED_SONAME_RELEASE "libdnnl.so.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS DNNL::dnnl )
list(APPEND _IMPORT_CHECK_FILES_FOR_DNNL::dnnl "${_IMPORT_PREFIX}/lib/libdnnl.so.1.7" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
