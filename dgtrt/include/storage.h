#ifndef DG_STORAGE_H
#define DG_STORAGE_H
#include <map>
#include <memory>

#define DLL_PUBLIC __attribute__ ((visibility("default")))
namespace dg
{

DLL_PUBLIC extern int add_request_storage(std::shared_ptr<void> req);
DLL_PUBLIC extern void* get_request_storage(int id);
DLL_PUBLIC extern void pop_request_storage(int id);
DLL_PUBLIC extern void enable_request_storage();
DLL_PUBLIC extern bool is_request_storage_enabled();
}; // namespace dg
#endif