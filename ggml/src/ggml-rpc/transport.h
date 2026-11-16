#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

// in the methods that take timeout_ms, it works as in poll(): a negative value waits without a limit, 0 does not wait
struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size, int timeout_ms = -1);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();
    bool local_endpoint(std::string & host, int & port) const;
    bool remote_endpoint(std::string & host, int & port) const;
    bool set_keepalive();

    socket_ptr accept(int timeout_ms = -1);

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);
    const char * transport_name() const;

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port, int timeout_ms = -1);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
