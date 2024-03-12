/*
 * flowcounter.{cc,hh} -- remove insults in web pages
 * Tom Barbette
 */

#include <click/config.h>
#include <click/router.hh>
#include <click/args.hh>
#include <click/error.hh>
#include "iportscandetector.hh"

CLICK_DECLS

IPortScanDetector::IPortScanDetector()
{

}

IPortScanDetector::~IPortScanDetector(){

}

int IPortScanDetector::configure(Vector<String> &conf, ErrorHandler *errh)
{
    Args args(conf, this, errh);

    if (parse(&args) || args
        .read_or_set("LIMIT", _limit, 10)
        .complete() < 0
        )
        return -1;

    return 0;
}

uint32_t IPortScanDetector::calculate_hash(Packet* packet){
        const uint32_t *srcIP = reinterpret_cast<const uint32_t*>(&(packet->ip_header()->ip_src));
        return rte_hash_crc_4byte(*srcIP, 0);
 }

inline void IPortScanDetector::process(IPSDState* state, Packet* p){
    uint32_t port = p->udp_header()->uh_dport;
    if(_shared)
        std::lock_guard<std::mutex> lock(mtx);
    
    if(state->count == 0){
        state->ports = new uint32_t[_limit];
        state->ports[0] = port;
        state->count++;
        return;
    }

    for(int i=0; i< state->count; i++){
        if(state->ports[state->count] == port)
            return;
    }

    if (state->count > _limit) return;

    state->ports[state->count] = port;
    state->count++;
}


CLICK_ENDDECLS
EXPORT_ELEMENT(IPortScanDetector)
ELEMENT_MT_SAFE(IPortScanDetector)
