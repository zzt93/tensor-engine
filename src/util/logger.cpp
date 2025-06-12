#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <mutex>

#include "../../include/util.h"
#include "../../include/enum.h"

using namespace tensorengine;

class Logger {
private:
    LogLevel level;
    std::ofstream ofs;
    std::mutex mtx;

    static std::string getCurrentTime() {
        auto t = std::time(nullptr);
        char buf[20];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        return buf;
    }

    static std::string levelToString(LogLevel lvl) {
        switch (lvl) {
            case LogLevel::DEBUG:
                return "DEBUG";
            case LogLevel::INFO:
                return "INFO";
            case LogLevel::WARNING:
                return "WARNING";
            case LogLevel::ERROR:
                return "ERROR";
            default:
                return "UNKNOWN";
        }
    }

public:
    Logger(LogLevel lvl = LogLevel::INFO, const std::string &filename = "") : level(lvl) {
        if (!filename.empty()) {
            ofs.open(filename, std::ios::app);
        }
    }

    ~Logger() {
        if (ofs.is_open()) {
            ofs.close();
        }
    }

    void setLevel(LogLevel lvl) {
        level = lvl;
    }

    void log(LogLevel msgLevel, const std::string &msg) {
        if (msgLevel < level) return;
        std::string timeStr = getCurrentTime();
        std::string levelStr = levelToString(msgLevel);
        std::string fullMsg = "[" + timeStr + "] " + levelStr + ": " + msg;

        std::lock_guard<std::mutex> lock(mtx);
        if (ofs.is_open()) {
            ofs << fullMsg << std::endl;
        } else {
            std::cout << fullMsg << std::endl;
        }
    }
};