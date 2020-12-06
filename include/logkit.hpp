/**
 * 测试log记录方法，将流导入到文件、输入输出流等中，同时使用log等级
*/
#ifndef BRIXLAB_LOGKIT_TOOLS_
#define BRIXLAB_LOGKIT_TOOLS_
#include <iostream>
#include <streambuf>
#include <sstream>
#include "stdlib.h"
#include <stdexcept>
using namespace std;
enum LogLevel{
    DEBUG_INFO = 0,
    FATAL_ERROR = 1,
};

inline std::string get_log_info(LogLevel level){
    std::string result;
    switch (level)
    {
    case LogLevel::DEBUG_INFO:
        result = "DEBUG_INFO";
        break;
    case LogLevel::FATAL_ERROR:
        result = "FATAL_ERROR";
        break;
    default:
        break;
    }
    return result;
}

class LogStreamBuf : public std::streambuf {
    public:
    // REQUIREMENTS: "len" must be >= 1 to account for the '\n' and '\0'.
    LogStreamBuf(char *buf, int len) {
        setp(buf, buf + len - 2);
    }

    // This effectively ignores overflow.
    virtual int_type overflow(int_type ch) {
        return ch;
    }

    // Legacy public ostrstream method.
    size_t pcount() const { return pptr() - pbase(); }
    char* pbase() const { return std::streambuf::pbase(); }
};

class LogMessage{
    public:
    class Logstream: public std::ostream{
        public:
            Logstream(char *buf, int len, int ctr):std::ostream(NULL),
                streambuf_(buf, len),
                ctr_(ctr),
                self_(this) {
                rdbuf(&streambuf_);                   
            }

            int ctr() const { return ctr_; }
            void set_ctr(int ctr) { ctr_ = ctr; }
            Logstream* self() const { return self_; }

            // Legacy std::streambuf methods.
            size_t pcount() const { return streambuf_.pcount(); }
            char* pbase() const { return streambuf_.pbase(); }
            char* str() const { return pbase(); }

        private:
            Logstream(const Logstream&);
            Logstream& operator=(const Logstream&);
            LogStreamBuf streambuf_;
            int ctr_;  // Counter hack (for the LOG_EVERY_X() macro)
            Logstream *self_;  // Consistency check hack
    };
    LogMessage(std::string file, std::string func, int line_, LogLevel level_, std::string task):
        filename(file), function(func), line(line_), level(level_){
            data = new (&thread_msg_data) LogMessageData;
            std::string loglevel = get_log_info(level);
            stream().fill('0');
            stream()<<"["<<loglevel<<"]"<<"["<<task<<"]"<< "["<<filename<<"]["<<function<<"]["<<line<<"]: ";
            data->num_char_to_log_ = 0;
            data->has_been_flushed_ =false;

    }
    LogMessage(){
        data = new (&thread_msg_data) LogMessageData;      
    }
    struct LogMessageData
    {
        LogMessageData(): stream_(message_text_, 200, 0){

        }
        Logstream stream_;
        char message_text_[200];
        int num_char_to_log_;
        bool has_been_flushed_;
    };
    
    ostream& stream(){
        return data->stream_;
    }
    
    void Flush(){
        if(data->has_been_flushed_)
            return;
        bool append_newline = data->message_text_[data->stream_.pcount() - 1] != '\n';
        data->num_char_to_log_ = data->stream_.pcount();
        if(append_newline){
            data->message_text_[data->num_char_to_log_++] = '\n';
        }
        //printf("message: %s", data->message_text_);
        send_log();
        if(append_newline){
            data->message_text_[data->num_char_to_log_-1] = '\0';
        }
        data->has_been_flushed_ = true;
    }

    ~LogMessage(){
        Flush();
    }

    void* send_log(){
        if(level == FATAL_ERROR){
            fprintf(stderr, "\033[0;3%sm", "");
            fwrite(data->message_text_, data->num_char_to_log_, 1, stderr);
            fprintf(stderr, "\033[m");
            exit(0);
        }else if(level == DEBUG_INFO){
            fprintf(stderr, "\033[0;3%sm", "");
            fwrite(data->message_text_, data->num_char_to_log_, 1, stderr);
            fprintf(stderr, "\033[m");
        }
    }
    private:
    std::aligned_storage<sizeof(LogMessage::LogMessageData),
                        alignof(LogMessage::LogMessageData)>::type thread_msg_data;
    std::string filename;
    std::string function;
    int line;
    LogMessageData* data;
    LogLevel level;
};

#define COMPACT_LOG_DEBUG_INFO(task) LogMessage(__FILE__, __FUNCTION__,__LINE__, DEBUG_INFO, task)
#define COMPACT_LOG_FATAL_ERROR(task) LogMessage(__FILE__, __FUNCTION__,__LINE__, FATAL_ERROR, task)
#define LOG(mode, task)  COMPACT_LOG_##mode(task).stream()

#define LOG_CHECK(expression, task)\
        if(!(expression))\
            LOG(FATAL_ERROR, task)
//目前只支持LOG(DEBUG_INFO), LOG(FATAL_ERROR), LOG_CHECK(expression)
#endif