/*
 * flowcounter.{cc,hh} -- remove insults in web pages
 * Tom Barbette
 */

#include <click/config.h>
#include <click/router.hh>
#include <click/args.hh>
#include <click/error.hh>
#include "ipolicer.hh"

CLICK_DECLS

IPolicer::IPolicer()
{

}

IPolicer::~IPolicer(){

}

int IPolicer::configure(Vector<String> &conf, ErrorHandler *errh)
{
    Args args(conf, this, errh);

    if (parse(&args) || args
        .read_or_set("MODE", _mode, 0)
        .complete() < 0
        )
        return -1;

    return 0;
}

uint32_t IPolicer::calculate_hash(Packet* packet){
    const uint32_t *dstIP = reinterpret_cast<const uint32_t*>(&(packet->ip_header()->ip_dst));
    return rte_hash_crc_4byte(*dstIP, 0);
}

inline void IPolicer::process(IPolicerState* state, Packet* p){
//    if(_shared)
//        std::lock_guard<std::mutex> lock(mtx);
    state->count++;
}


CLICK_ENDDECLS
EXPORT_ELEMENT(IPolicer)
ELEMENT_MT_SAFE(IPolicer)
