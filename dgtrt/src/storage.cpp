#include "storage.h"
#include <mutex>

namespace dg
{

    class DgStorage
    {
    public:
        DgStorage() {}

        void enable(bool do_enable)
        {
            enabled_ = do_enable;
        }
        bool enabled()
        {
            return enabled_;
        }

        int add(std::shared_ptr<void> req)
        {
            std::unique_lock<std::mutex> lock;
            auto id = ++seq_;
            requests_[id] = req;
            return id;
        }

        void *get(int id)
        {
            std::unique_lock<std::mutex> lock;
            auto it = requests_.find(id);
            if (it != requests_.end())
            {
                return it->second.get();
            }
            return nullptr;
        }

        void pop(int id)
        {
            std::unique_lock<std::mutex> lock;
            auto it = requests_.find(id);
            if (it != requests_.end())
            {
                requests_.erase(id);
            }
        }

    protected:
        std::map<int, std::shared_ptr<void>> requests_;
        bool enabled_ = false;
        std::mutex mt_;
        int seq_ = 0;
    };

    static DgStorage g_storage_;

    bool is_request_storage_enabled()
    {
        return g_storage_.enabled();
    }
    void enable_request_storage()
    {
        g_storage_.enable(true);
    }
    int add_request_storage(std::shared_ptr<void> req)
    {
        if (is_request_storage_enabled())
        {
            return g_storage_.add(req);
        }
        return -1;
    }

    void *get_request_storage(int id)
    {
        if (is_request_storage_enabled())
        {
            return g_storage_.get(id);
        }
        return nullptr;
    }

    void pop_request_storage(int id)
    {
        if (is_request_storage_enabled())
        {
            g_storage_.pop(id);
        }
    }
}; // namespace dg