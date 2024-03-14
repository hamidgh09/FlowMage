define($nfport 0000:17:00.0)

    fd0 :: FromDPDKDevice($nfport, HASH_FUNC FLOW,SCALE SHARE, PROMISC true, PAUSE none, MAXTHREADS 8, N_QUEUES 8, NUMA 1, VERBOSE 99, NDESC 4096, BURST 64, RSS_AGGREGATE false)
    -> Classifier(12/0800)
    -> MarkIPHeader(OFFSET 14)
    -> psd :: IPortScanDetector(SIZE 128000000, SHARED 1)
    -> dstCounter :: IFlowCounter(SIZE 128000000, SHARED 1, MODE 2)
    -> flowCounter :: IFlowCounter(SIZE 128000000, SHARED 1, MODE 0)

    -> td0 :: ToDPDKDevice($nfport, VERBOSE 99, TIMEOUT -1 , BLOCKING false);


//Script that runs every seconds
Script(TYPE ACTIVE,
        set use 0,
        label loop,
        set t $(now),

        set nuse $(add $(useful_kcycles)),
        set diff $(sub $nuse $use),
        set use $nuse,

        set npkts0 $(fd0.xstats rx_q0_packets),
        set cdiff0 $(sub $npkts0 $pkts0),
        set pkts0 $npkts0,

        set npkts1 $(fd0.xstats rx_q1_packets),
        set cdiff1 $(sub $npkts1 $pkts1),
        set pkts1 $npkts1,

        set npkts2 $(fd0.xstats rx_q2_packets),
        set cdiff2 $(sub $npkts2 $pkts2),
        set pkts2 $npkts2,

        set npkts3 $(fd0.xstats rx_q3_packets),
        set cdiff3 $(sub $npkts3 $pkts3),
        set pkts3 $npkts3,

        set npkts4 $(fd0.xstats rx_q4_packets),
        set cdiff4 $(sub $npkts4 $pkts4),
        set pkts4 $npkts4,

        set npkts5 $(fd0.xstats rx_q5_packets),
        set cdiff5 $(sub $npkts5 $pkts5),
        set pkts5 $npkts5,

        set npkts6 $(fd0.xstats rx_q6_packets),
        set cdiff6 $(sub $npkts6 $pkts6),
        set pkts6 $npkts6,

        set npkts7 $(fd0.xstats rx_q7_packets),
        set cdiff7 $(sub $npkts7 $pkts7),
        set pkts7 $npkts7,

        set pktsmin $(min $cdiff0 $cdiff1 $cdiff2 $cdiff3 $cdiff4 $cdiff5 $cdiff6 $cdiff7),
        set pktsmax $(max $cdiff0 $cdiff1 $cdiff2 $cdiff3 $cdiff4 $cdiff5 $cdiff6 $cdiff7),
        set pktimbalance $(div $pktsmax $pktsmin),

        print "PKTS "$cdiff0 $cdiff1 $cdiff2 $cdiff3 $cdiff4 $cdiff5 $cdiff6 $cdiff7,
        print "PACKET_IMBALANCE_PERCENTAGE " $(mul $pktimbalance 100),
        print "FLOW_COUNT: " $(flowCounter.count),
        read load,
        write c.reset,
        wait 1s,
        gotoa loop);

DriverManager(
                read fd0.rss_reta,
                wait,
                read fd0.xstats,
                print "DUMP: "$(man.dump),
                print "COUNT: "$(man.count),
                print "CONFLICT: "$(man.conflict),
                print "RESULT-BATCH "$(bs.dump),
                print "RESULT-COUNT-SC "$(sc.count),
                print "RESULT-INSERTIONS-SC "$(sc.insertions),
                print "RESULT-COUNT-DC "$(dc.count),
                print "RESULT-INSERTIONS-DC "$(dc.insertions),
                print "RESULT-NFFLOWS "$(man.count),
                print "RESULT-FAILED-SEARCHES "$(man.failed_searches),
                print "RESULT-SUCCESSFUL-SEARCHES "$(man.successful_searches),
                print "RESULT-NFRXCOUNT "$(fd0.hw_count),
                print "RESULT-NFPHYCOUNT "$(fd0.xstats rx_phy_packets),
                print "RESULT-NFPHYDROPPED "$(fd0.xstats rx_phy_discard_packets),
                print "RESULT-NFRXDROPPED "$(fd0.hw_dropped),
                print "RESULT-NFTWCOUNT "$(td0.hw_count),
                print "RESULT-NFCYCLESPP "$(div $(mul $(add $(useful_kcycles)) 1000) $(fd0.count)),
                print "RESULT-REF_BUA $(bua.average)",
                print "RESULT-REF_BUB $(bub.average)",
                print "RESULT-REF_BU0 $(ref/bu0.average)",
                print "RESULT-REF_BU1 $(ref/bu1.average)",
                print "RESULT-REF_BU2 $(ref/bu2.average)",
                print "RESULT-REF_BU3 $(ref/bu3.average)",
                print "RESULT-REF_BU4 $(ref/bu4.average)",
                print "RESULT-REF_BU5 $(ref/bu5.average)",
                print "RESULT-REF_BU6 $(ref/bu6.average)",
                print "RESULT-REF_BU7 $(ref/bu7.average)",
                print "RESULT-REF_BYPASSED $(ref/bypassc.count)",
                print "RESULT-REF_SERVED   $(served.count)",
                print "RESULT-REF_AFTER_C  $(afterRef.count)",
                print "RESULT-REF_LARGES   $(ref/lcounter.count)",
                print "RESULT-MERGE_SIZE   $(bmerger.avg_merge_size)",
                print "RESULT-CACHE-FAILED-SEARCHES "$(cache.failed_searches),
                print "RESULT-CACHE-SUCCESSFUL-SEARCHES "$(cache.successful_searches),
                print "RESULT-CACHE-SUCCESS-RATE "$(cache.success_rate),
                print "RESULT-FC-COUNT: $(fc.idump)",
                print "END"
                );