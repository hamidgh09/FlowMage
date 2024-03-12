/*
 * flowcounter.{cc,hh} -- remove insults in web pages
 * Tom Barbette
 */

#include <click/config.h>
#include <click/router.hh>
#include <click/args.hh>
#include <click/error.hh>
#include "isourcecounter.hh"

CLICK_DECLS

ISourceCounter::ISourceCounter()
{

}

ISourceCounter::~ISourceCounter(){

}

int ISourceCounter::configure(Vector<String> &conf, ErrorHandler *errh)
{
    Args args(conf, this, errh);

    if (parse(&args) || args
        .read_or_set("MODE", _mode, 0)
        .complete() < 0
        )
        return -1;

    return 0;
}

uint32_t ISourceCounter::calculate_hash(Packet* packet){
    const uint32_t *srcIP = reinterpret_cast<const uint32_t*>(&(packet->ip_header()->ip_src));
    return rte_hash_crc_4byte(*srcIP, 0);
}

inline void ISourceCounter::process(ISourceCounterState* state, Packet* p){
//    if(_shared)
//        std::lock_guard<std::mutex> lock(mtx);
    state->count++;
}


CLICK_ENDDECLS
EXPORT_ELEMENT(ISourceCounter)
ELEMENT_MT_SAFE(ISourceCounter)
