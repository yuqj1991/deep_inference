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

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

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
    LogMessage(std::string file, std::string func, int line_, LogLevel level_):
        filename(file), function(func), line(line_), level(level_){
            data = new (&thread_msg_data) LogMessageData;
            std::string loglevel = get_log_info(level);
            stream().fill('0');
            stream()<<"["<<loglevel<<"]"<< "["<<filename<<"]["<<function<<"]["<<line<<"]: ";
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
        send_log();
        if(append_newline){
            data->message_text_[data->num_char_to_log_-1] = '\0';
        }
        data->has_been_flushed_ = true;
    }

    ~LogMessage(){
        Flush();
    }

    void send_log(){
        if(level == FATAL_ERROR){
            fprintf(stderr, "\033[0;31m");
            fwrite(data->message_text_, data->num_char_to_log_, 1, stderr);
            exit(0);
        }else if(level == DEBUG_INFO){
            fprintf(stderr, "\033[0;33m");
            fwrite(data->message_text_, data->num_char_to_log_, 1, stderr);
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

#define COMPACT_LOG_DEBUG_INFO LogMessage(__FILE__, __FUNCTION__,__LINE__, DEBUG_INFO)
#define COMPACT_LOG_FATAL_ERROR LogMessage(__FILE__, __FUNCTION__,__LINE__, FATAL_ERROR)
#define LOG(mode)  COMPACT_LOG_##mode.stream()

#define LOG_CHECK(expression)\
        if(!(expression))\
            LOG(FATAL_ERROR)
//目前只支持LOG(DEBUG_INFO), LOG(FATAL_ERROR), LOG_CHECK(expression)
#endif