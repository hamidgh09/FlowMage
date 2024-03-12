#ifndef CLICK_ISOURCECOUNTER_HH
#define CLICK_ISOURCECOUNTER_HH
#include <click/element.hh>
#include <click/vector.hh>
#include <click/multithread.hh>
#include <mutex>

#include "iflowmanager.hh"

CLICK_DECLS

/*
=c

FlowCounter([CLOSECONNECTION])

=s flow

Counts all flows passing by, the number of active flows, and the number of 
packets per flow.

 */


class ISourceCounterState : public IState {

};

class ISourceCounter : public IFlowManager<ISourceCounterState>
{
public:
    /** @brief Construct an FlowCounter element
     */
    ISourceCounter() CLICK_COLD;
    ~ISourceCounter() CLICK_COLD;

    const char *class_name() const override        { return "ISourceCounter"; }
    const char *port_count() const override        { return PORTS_1_1; }
    const char *processing() const override        { return PUSH; }
    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;

    inline uint32_t calculate_hash(Packet* packet) override CLICK_COLD;
    inline void process(ISourceCounterState* , Packet*) override;

    private:
    std::mutex mtx;
    uint8_t _mode;
};

CLICK_ENDDECLS
#endif
