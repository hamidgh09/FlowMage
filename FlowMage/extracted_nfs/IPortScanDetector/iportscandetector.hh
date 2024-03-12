#ifndef CLICK_IPORTSCANDETECTOR_HH
#define CLICK_IPORTSCANDETECTOR_HH
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


class IPSDState : public IState {
    public:
        uint32_t *ports;

};

class IPortScanDetector : public IFlowManager<IPSDState>
{
public:
    /** @brief Construct an FlowCounter element
     */
    IPortScanDetector() CLICK_COLD;
    ~IPortScanDetector() CLICK_COLD;

    const char *class_name() const override        { return "IPortScanDetector"; }
    const char *port_count() const override        { return PORTS_1_1; }
    const char *processing() const override        { return PUSH; }
    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;

    inline uint32_t calculate_hash(Packet* packet) override CLICK_COLD;
    inline void process(IPSDState* , Packet*) override;

    private:
    std::mutex mtx;
    uint8_t _limit;
};

CLICK_ENDDECLS
#endif
