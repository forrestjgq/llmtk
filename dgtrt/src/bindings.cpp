#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "storage.h"

namespace py = pybind11;
namespace dg {
#define CLASS(cls, doc)            py::class_<cls, Ptr<cls>>(m, #cls, doc)
#define SUBCLASS(cls, parent, doc) py::class_<cls, parent, Ptr<cls>>(m, #cls, doc)

// template <typename T>
// std::shared_ptr<T> copy_array(py::array_t<T>& arr, int* sz)
// {
//     auto buf = arr.request();
//     assert(buf.itemsize == sizeof(T));
//     auto ptr = std::shared_ptr<T>(new T[buf.shape[0]]);
//     memcpy(ptr.get(), buf.ptr, buf.size * buf.itemsize);
//     *sz = buf.size;
//     return ptr;
// }

// int py_add_request_storage(py::array_t<int> input_ids, std::map<int, py::array_t<DataType>>& images)
// {
//     auto req = std::make_shared<Request>();
//     req->input_ids = copy_array<int>(input_ids, &req->input_ids_len);

//     req->image_len = 0;
//     int sz;
//     for (auto it = images.begin(); it != images.end(); ++it)
//     {
//         req->images[it->first] = copy_array<DataType>(it->second, &sz);
//         assert(sz > 0);
//         if (req->image_len == 0)
//         {
//             req->image_len = sz;
//         }
//         else
//         {
//             assert(req->image_len == sz);
//         }
//     }
//     return add_request_storage(req);
// }
std::shared_ptr<void> copy_array(py::array& arr)
{
    auto buf = arr.request();
    int sz = buf.size * buf.itemsize;
    auto ptr = new uint8_t[sz];
    memcpy(ptr, buf.ptr, sz);
    return std::shared_ptr<void>(ptr);
}

int py_add_request_storage(py::array image)
{
    return add_request_storage(copy_array(image));
}

PYBIND11_MODULE(bindings, m) {
    m.def("add_request_storage", &py_add_request_storage, "add request storage to system for lookup plugins");
    m.def("remove_request_storage", &pop_request_storage, "pop out request storage by id");
    m.def("enable_request_storage", &enable_request_storage, "enable request storage");
}

};  // namespace 