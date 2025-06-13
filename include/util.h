#pragma once

#include "enum.h"
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <mutex>
#include <iostream>
#include <set>
#include <map>
#include <string>
#include <sstream>
#include <type_traits>
#include <queue>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <list>


namespace tensorengine {

    template<typename T>
    class BlockingQueue {
    private:
        struct ListNode {
            T t;
            ListNode* next = nullptr;
            ListNode() {}
            explicit ListNode(const T& value) : t(value), next(nullptr) {}
        };
        std::mutex head_;
        std::mutex tail_;
        std::condition_variable elem_cond_;
        std::atomic<bool> stop_;

        ListNode* h_;
        ListNode* t_;
        std::atomic<int> size_;

    public:
        BlockingQueue(){
            h_ = new ListNode();
            t_ = h_;
        }
        BlockingQueue(const BlockingQueue&) = delete;
        BlockingQueue& operator=(const BlockingQueue&) = delete;
        ~BlockingQueue() {
            while (h_ != nullptr) {
                ListNode* temp = h_;
                h_ = h_->next;
                delete temp;
            }
        }

        bool empty() {
            return size_==0;
        }

        T pop() {
            std::unique_lock<std::mutex> lock(head_);
            elem_cond_.wait(lock, [this]{ return !this->empty() || stop_; });
            if (stop_) {
                throw std::runtime_error("waiting for data interrupted");
            }
            auto node = h_->next;
            assert(node != nullptr);
            h_->next = node->next;
            auto t = std::move(node->t);
            delete node;
            size_--;
            return t;
        }

        void push_back(const T& t) {
            auto node = new ListNode(t);
            {
                std::lock_guard<std::mutex> lock(tail_);
                t_->next = node;
                t_ = node;
                size_++;
                elem_cond_.notify_one();
            }
        }

        void stop() {
            stop_ = false;
            elem_cond_.notify_all();
        }

    };

    template<typename T, typename _Compare=std::less<T>>
    class ConcurrentPriorityQueue {

    public:
        ConcurrentPriorityQueue() = default;
        ConcurrentPriorityQueue(const ConcurrentPriorityQueue&) = delete;
        ConcurrentPriorityQueue& operator=(const ConcurrentPriorityQueue&) = delete;
    };

    template<typename K, typename V, typename _Hash = std::hash<K>, typename _Pred = std::equal_to<K>>
    class ConcurrentHashMap {
    private:
        static const size_t DEFAULT_BUCKET_COUNT = 16;
        static constexpr float LOAD_FACTOR = 0.75;
        typedef std::pair<K, V> KeyValuePair;
        typedef std::list<KeyValuePair> Bucket;


        std::vector<Bucket> bucket_;
        mutable std::vector<std::mutex> bucket_lock_;
        std::atomic<int> size_ = 0;
        std::atomic<int> capacity_ = DEFAULT_BUCKET_COUNT;
        _Hash hash_f_;
        _Pred equal_f_;

        void insert_unique(std::pair<const K, V> pair) {
            size_t i = idx(pair.first);
            std::lock_guard<std::mutex> lock(bucket_lock_[i]);
            if (bucket_[i].empty()) {
                bucket_[i].push_back(pair);
            } else {
                for (auto &item: bucket_[i]) {
                    if (equal_f_(item.first, pair.first)) {
                        item.second = pair.second;
                        return;
                    }
                }
                bucket_[i].push_back(pair);
            }
        }

        inline size_t idx(const K& k) const {
            size_t hash = hash_f_(k);
            size_t idx = hash & (capacity_ - 1);
            return idx;
        }

        void resize() {

        }
    public:
        class iterator {
            using buckets_type = const std::vector<Bucket>*;
            using bucket_iter_type = typename Bucket::const_iterator;

            buckets_type buckets_ptr;
            size_t bucket_index;
            bucket_iter_type bucket_iter;

            void move_to_next_valid() {
                // 假如当前迭代器已经指向本桶尾部，则找下一个非空桶
                while (bucket_index < buckets_ptr->size() && bucket_iter == (*buckets_ptr)[bucket_index].end()) {
                    ++bucket_index;
                    if (bucket_index < buckets_ptr->size()) {
                        bucket_iter = (*buckets_ptr)[bucket_index].begin();
                    }
                }
            }

        public:
            using pointer = const KeyValuePair*;
            using reference = const KeyValuePair&;

            iterator(buckets_type buckets_ptr_, size_t bucket_index_, bucket_iter_type bucket_iter_)
                    : buckets_ptr(buckets_ptr_), bucket_index(bucket_index_), bucket_iter(bucket_iter_) {
                if (bucket_index < buckets_ptr->size() && bucket_iter == (*buckets_ptr)[bucket_index].end()) {
                    move_to_next_valid();
                }
            }

            reference operator*() const { return *bucket_iter; }
            pointer operator->() const { return &(*bucket_iter); }

            iterator& operator++() {
                ++bucket_iter;
                if (bucket_iter == (*buckets_ptr)[bucket_index].end()) {
                    ++bucket_index;
                    if (bucket_index < buckets_ptr->size()) {
                        bucket_iter = (*buckets_ptr)[bucket_index].begin();
                        move_to_next_valid();
                    }
                }
                return *this;
            }

            iterator operator++(int) {
                iterator tmp = *this;
                ++*this;
                return tmp;
            }

            bool operator==(const iterator& other) const {
                return buckets_ptr == other.buckets_ptr && bucket_index == other.bucket_index && bucket_iter == other.bucket_iter;
            }

            bool operator!=(const iterator& other) const {
                return !(*this == other);
            }
        };
        
        ConcurrentHashMap(): bucket_(DEFAULT_BUCKET_COUNT), bucket_lock_(DEFAULT_BUCKET_COUNT), hash_f_(_Hash()), equal_f_(_Pred()) {}
        ConcurrentHashMap(const ConcurrentHashMap&) = delete;
        ConcurrentHashMap& operator=(const ConcurrentHashMap&) = delete;

        template <class _InputIterator>
        inline void insert(_InputIterator __first, _InputIterator __last) {
            for (; __first != __last; ++__first)
                insert_unique(*__first);
        }

        // 只能读，写并发不安全
        const V& operator[](const K& key) const {
            auto it = find(key);
            if (it == end()) {
                throw std::out_of_range("key not found");
            }
            return it->second;
        }

        void insert(const std::pair<K, V> p){
            insert_unique(p);
        }

        iterator find(const K& key) const {
            size_t i = idx(key);
            std::lock_guard<std::mutex> lock(bucket_lock_[i]);
            const Bucket &bucket = bucket_[i];
            if (!bucket.empty()) {
                for (auto it = bucket.begin(); it != bucket.end(); it++) {
                    if (equal_f_(it->first, key)) {
                        return iterator(&bucket_, bucket_.size(), it);
                    }
                }
            }
            return end();
        }


        iterator begin() {
            return iterator(&bucket_, 0, bucket_[0].begin());
        }
        iterator end() {
            return iterator(&bucket_, bucket_.size(), typename Bucket::iterator{});
        }

        iterator begin() const {
            return begin();
        }
        iterator end() const {
            return end();
        }

        size_t size() {
            return size_;
        }
        
        size_t capacity() {
            return capacity_;
        }

    };

    class ThreadPool {
    private:
        // 工作线程
        std::vector<std::thread> workers;
        // 任务队列
        BlockingQueue<std::function<void()>> tasks;

        std::atomic<bool> stop;
    public:
        explicit ThreadPool(size_t thread_count): stop(false){
            for (size_t i = 0; i < thread_count; ++i) {
                workers.emplace_back([this]() {
                    while (true) {
                        std::function<void()> task = tasks.pop();
                        task();
                    }
                });
            }
        }
        ~ThreadPool() {
            stop = true;
            tasks.stop();
            for (auto& worker: workers)
                worker.join();
        }

        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>
        {
            using return_type = typename std::invoke_result<F, Args...>::type;

            // 封装任务为 packaged_task
            auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            std::future<return_type> res = task->get_future();

            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.push_back([task]() { (*task)(); });
            return res;
        }
    };


    std::vector<float> rands(int limit, float min, float max);

    std::string tostring(DataType value);

    template<typename K, typename V>
    std::string tostring(const ConcurrentHashMap<K, V>& m) {
        std::ostringstream oss;
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : m) {
            if (!first) oss << ", ";
            oss << key << ": " << value;
            first = false;
        }
        oss << "}";
        return oss.str();
    }

    template<typename Container>
    std::string tostring(const Container& s) {
        std::ostringstream oss;
        oss << "{";
        bool first = true;
        for (const auto& elem : s) {
            if (!first) oss << ", ";
            oss << elem;
            first = false;
        }
        oss << "}";
        return oss.str();
    }

    class Logger {
    private:
        LogLevel level;
        std::ofstream ofs;
        std::mutex mtx;
        BlockingQueue<std::string> queue{};

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
        explicit Logger(LogLevel lvl = LogLevel::INFO, const std::string &filename = "") : level(lvl) {
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

        void logAsync(LogLevel msgLevel, const std::string &msg) {

        }
    };

}