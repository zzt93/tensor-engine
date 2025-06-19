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
#include <sstream>
#include "cassert"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(-1); \
        } \
    } while (0)


#define ASSERT_MSG(cond, msg_expr) \
    do { \
        if (!(cond)) { \
            std::ostringstream _oss; _oss << msg_expr; \
            std::fprintf(stderr, "Assertion failed: %s\nMessage: %s\nFile: %s:%d\n", \
                         #cond, _oss.str().c_str(), __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while (0)

namespace tensorengine {

// 2-3 tree, left red
// left < mid < right
    template<typename Key, typename T, class _Compare = std::less<Key>>
    class RBTree2 {
    private:
        class TreeNode {
        public:
            std::pair<Key, T> val;
            TreeNode *parent;
            TreeNode *left;
            TreeNode *right;
            Color c;

            TreeNode(std::pair<Key, T> data, TreeNode* parent): val(data), c(RED), parent(parent), left(nullptr), right(nullptr) {
            }

            void flipColor() {
                if (c == RED) c = BLACK;
                else c = RED;
            }

            void setLeft(TreeNode *l) {
                this->left = l;
                if (l == nullptr) return;
                l->parent = this;
            }

            void setRight(TreeNode *r) {
                this->right = r;
                if (r == nullptr) return;
                r->parent = this;
            }

            friend std::ostream& operator<<(std::ostream& os, const TreeNode* n) {
                os << n->colorFormat() << n->val.first << "\033[0m";
                return os;
            }

            std::string colorFormat() const {
                switch (this->c) {
                    case RED:
                        return "\033[31m";
                    case BLACK:
                        return "\033[30m";
                }
            }
        };

        TreeNode *fake;
        _Compare cmp;

        bool isLessThan(const Key& l, const Key& v) const {
            return cmp(l, v);
        }

        TreeNode *root() const {
            return fake->left;
        }

        void setRoot(TreeNode *r) {
            fake->setLeft(r);
        }

        void leftRotate(TreeNode *pivot) {
            auto parent = pivot->parent;
            auto right = pivot->right;
            assert(parent != nullptr);
            assert(right != nullptr);

            if (pivot == parent->left) {
                parent->setLeft(right);
            } else {
                parent->setRight(right);
            }
            auto subLeft = right->left;
            right->setLeft(pivot);
            pivot->setRight(subLeft);
        }

        void rightRotate(TreeNode *pivot) {
            auto parent = pivot->parent;
            auto left = pivot->left;
            assert(parent != nullptr);
            assert(left != nullptr);

            if (parent->left == pivot) {
                parent->setLeft(left);
            } else {
                parent->setRight(left);
            }
            auto subRight = left->right;
            left->setRight(pivot);
            pivot->setLeft(subRight);
        }

        void redNodeWithRedLeft(TreeNode *parent) {
            assert(parent->c == RED);
            assert(parent->left->c == RED);

            TreeNode *grand = parent->parent;
            assert(grand != nullptr);
            rightRotate(grand);
            parent->flipColor();
            grand->flipColor();

            flipUp(parent);
        }

        void flipUp(TreeNode *parent) {
            // B
            //R R
            assert(parent->left->c == RED);
            assert(parent->right->c == RED);
            bool twoSonRed = (parent->right != nullptr && parent->right->c == RED) && (parent->left != nullptr && parent->left->c == RED) && parent->c == BLACK;
            assert(twoSonRed);
            TreeNode* red = nullptr;
            parent->right->flipColor();
            parent->left->flipColor();
            parent->flipColor();
            if (parent == root()) {
                parent->flipColor();
                return;
            }
            red = parent;
            parent = parent->parent;
            twoSonRed = (parent->right != nullptr && parent->right->c == RED) && (parent->left != nullptr && parent->left->c == RED) && parent->c == BLACK;
            if (twoSonRed) {
                flipUp(parent);
                return;
            }
            bool twoRed = parent->c == RED;
            if (!twoRed) {
                if (red == parent->right) {
                    leftRotate(parent);
                    red->flipColor();
                    parent->flipColor();
                }
                return;
            }
            //  R
            // R *
            if (red == parent->right) {
                // 处理右倾
                leftRotate(parent);

                redNodeWithRedLeft(red);
            } else {
                redNodeWithRedLeft(parent);
            }
        }

        // return if exist
        bool binarySearch(TreeNode *r, Key v, TreeNode*& insertionNode) const {
            if (r == nullptr) {
                return false;
            }
            bool last = false;
            if (isLessThan(r->val.first, v)) {
                if (r->right != nullptr) {
                    return binarySearch(r->right, v, insertionNode);
                } else {
                    last = true;
                }
            } else if (isLessThan(v, r->val.first)) {
                if (r->left != nullptr) {
                    binarySearch(r->left, v, insertionNode);
                } else {
                    last = true;
                }
            } else {
                insertionNode = r;
                return true;
            }
            if (last) {
                insertionNode = r;
            }
            return false;
        }

        TreeNode* pre(TreeNode* now) {
            assert(false);
        }

        static TreeNode* min(TreeNode* now) {
            if (now == nullptr) {
                return nullptr;
            }
            while (now->left != nullptr) {
                now = now->left;
            }
            return now;
        }

        static TreeNode* next(TreeNode* now, TreeNode* root) {
            if (now == nullptr) assert(false);
            if (now->right != nullptr) {
                return min(now->right);
            }
            while (now != root) {
                TreeNode* parent = now->parent;
                if (parent->left == now) {
                    return parent;
                }
                now = parent;
            }
            return nullptr;
        }
    public:
        template<typename Pointer, typename Reference>
        class _iterator_base {
        protected:
            TreeNode *now;
            TreeNode *root;
        public:
            _iterator_base(TreeNode* root, TreeNode *n) : now(n), root(root) {}

        public:

            Reference operator*() const { return now->val; }
            Pointer operator->() const { return &(now->val); }

            _iterator_base& operator++() {
                now = next(now, root);
                return *this;
            }

            _iterator_base operator++(int) {
                iterator tmp = *this;
                ++*this;
                return tmp;
            }

            bool operator==(const _iterator_base& other) const {
                return now == other.now;
            }

            bool operator!=(const _iterator_base& other) const {
                return !(*this == other);
            }

        };

        class iterator : public _iterator_base<std::pair<Key, T>*, std::pair<Key, T>&> {
        public:
            using base = _iterator_base<std::pair<Key, T>*, std::pair<Key, T>&>;
            using base::base;
        };

        class const_iterator : public _iterator_base<const std::pair<Key, T>*, const std::pair<Key, T>&> {
        public:
            using base = _iterator_base<const std::pair<Key, T>*, const std::pair<Key, T>&>;
            using base::base;
        };

        iterator begin() {
            return iterator(root(), min(root()));
        }
        iterator end() {
            return iterator(root(), nullptr);
        }
        const_iterator begin() const {
            return const_iterator(root(), min(root()));
        }
        const_iterator end() const {
            return const_iterator(root(), nullptr);
        }


        RBTree2(): fake(new TreeNode(make_pair(Key{}, T{}), nullptr)), cmp(_Compare()) {
        }

        friend std::ostream& operator<<(std::ostream& os, const RBTree2<Key, T>& p) {
            os << "RBTree[" << std::endl;
            if (p.root() == nullptr) {
                os << "]";
                return os;
            }
            TreeNode empty(make_pair(Key{}, T{}), nullptr);
            TreeNode blackLeaf(make_pair(Key{}, T{}), nullptr);
            blackLeaf.flipColor();

            std::queue<TreeNode*> q;
            q.push(p.root());
            q.push(&empty);
            while (!q.empty()) {
                TreeNode* a = q.front();
                q.pop();
                if (a == &empty) {
                    os << std::endl;
                    if (!q.empty()) {
                        q.push(&empty);
                    }
                    continue;
                } else if (a == &blackLeaf) {
                    os << blackLeaf.colorFormat() << "null\033[0m ";
                    continue;
                }
                auto n = next(a, p.root());
                os << a << "(" << (n != nullptr ? std::to_string(n->val.first) : "nop") << ") ";
                if (a->left != nullptr) {
                    q.push(a->left);
                } else {
                    q.push(&blackLeaf);
                }
                if (a->right != nullptr) {
                    q.push(a->right);
                } else {
                    q.push(&blackLeaf);
                }
            }
            os << "]";
            return os;
        }

        void insert(Key k, T v) {
            if (this->root() == nullptr) {
                this->setRoot(new TreeNode(make_pair(k, v), nullptr));
                this->root()->flipColor();
                return;
            }
            TreeNode* parent = nullptr;
            bool found = binarySearch(this->root(), k, parent);
            if (found) {
                parent->val = make_pair(k, v);
                return;
            }
            if (isLessThan(k, parent->val.first)) {
                assert(parent->left == nullptr);
                parent->left = new TreeNode(make_pair(k, v), parent);
                if (parent->c == BLACK) {
                    return;
                }
                redNodeWithRedLeft(parent);
            } else {
                assert(parent->right == nullptr);
                auto now = new TreeNode(make_pair(k, v), parent);
                parent->right = now;

                if (parent->c == BLACK) {
                    if (parent->left == nullptr || parent->left->c == BLACK) {
                        // 处理右倾
                        leftRotate(parent);
                        parent->flipColor();
                        now->flipColor();
                    } else {
                        flipUp(parent);
                    }
                } else {
                    leftRotate(parent);
                    redNodeWithRedLeft(now);
                }
            }

        }

        iterator find(Key k) {
            TreeNode* parent = nullptr;
            bool exist = binarySearch(root(), k, parent);
            if (exist) {
                return iterator(root(), parent);
            }
            return end();
        }

        const_iterator find(Key k) const {
            TreeNode* parent = nullptr;
            bool exist = binarySearch(root(), k, parent);
            if (exist) {
                return const_iterator(root(), parent);
            }
            return end();
        }

        void push_back(std::pair<Key, T> p) {
            insert(p.first, p.second);
        }

        bool empty() {
            return root() == nullptr;
        }
    };

    template<typename T>
    class BlockingQueue {
    private:
        struct ListNode {
            T t;
            ListNode* next = nullptr;
            ListNode() {}
            explicit ListNode(const T& value) : t(value), next(nullptr) {}
        };
        std::mutex head_m_;
        std::mutex tail_m_;
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
            std::unique_lock<std::mutex> lock(head_m_);
            elem_cond_.wait(lock, [this]{ return !this->empty() || stop_; });
            if (stop_) {
                throw std::runtime_error("waiting for data interrupted");
            }
            auto node = h_->next;
            ASSERT_MSG(node != nullptr, "empty queue");
            h_->next = node->next;
            if (t_ == node) {
                // 如果弹出的是最后一个节点
                t_ = h_; // tail 重新指向 dummy head
            }
            auto t = std::move(node->t);
            delete node;
            size_--;
            return t;
        }

        void push_back(const T& t) {
            auto node = new ListNode(t);
            {
                std::lock_guard<std::mutex> lock(tail_m_);
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

        void error(const std::string &msg) {
            log(LogLevel::ERROR, msg);
        }
        void warn(const std::string &msg) {
            log(LogLevel::WARNING, msg);
        }
        void info(const std::string &msg) {
            log(LogLevel::INFO, msg);
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
            this->log(msgLevel, msg);
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
        typedef RBTree2<K, V> Bucket;


        std::vector<Bucket> bucket_;
        mutable std::vector<std::mutex> bucket_lock_;
        std::atomic<int> size_ = 0;
        std::atomic<int> capacity_ = DEFAULT_BUCKET_COUNT;
        _Hash hash_f_;
        _Pred equal_f_;

        void insert_unique(std::pair<const K, V> pair) {
            size_t i = idx(pair.first);
            std::lock_guard<std::mutex> lock(bucket_lock_[i]);
            auto it = bucket_[i].find(pair.first);
            if (it != bucket_[i].end()) {
                it->second = pair.second;
                return;
            } else {
                // it.insert?
                bucket_[i].push_back(pair);
                size_++;
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
            auto it = bucket.find(key);
            if (it != bucket.end()) {
                return iterator(&bucket_, bucket_.size(), it);
            }
            return end();
        }


        iterator begin() {
            return iterator(&bucket_, 0, bucket_[0].begin());
        }
        iterator end() {
            return iterator(&bucket_, size(), typename Bucket::const_iterator{nullptr, nullptr});
        }
        using const_iterator = iterator;
        const_iterator begin() const {
            return const_iterator(&bucket_, 0, bucket_[0].begin());
        }
        const_iterator end() const {
            return const_iterator(&bucket_, size(), typename Bucket::const_iterator{nullptr, nullptr});
        }

        size_t size() const {
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
        inline static Logger logger{};
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
            tasks.push_back([task, this]() {
                try {
                    (*task)();
                } catch (...) {
                    logger.error("exec met exception");
                    stop = true;
                    // 这部分可以省略，因为 packaged_task 已自动处理
                }
            });

            return res;
        }

        bool stopped() {
            return stop;
        }
    };

    template <typename T>
    size_t setHash(const std::set<T>& s) {
        size_t seed = 0;
        // 合并集合中所有元素的哈希值
        for (const auto& elem : s) {
            seed ^= std::hash<T>{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }

    std::vector<float> rands(int limit, float min, float max);

    std::string tostring(DataType value);
    std::string tostring(DeviceType value);

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


}