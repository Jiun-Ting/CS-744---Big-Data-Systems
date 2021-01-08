# Courseworks and final report of Big Data Systems

* Apache Spark Deployment and Testing: Deployed Hadoop and Spand on Cloudlab and conducted experiments to observe the result of RDD partitioning, RDD persistence and fault tolerance by implementing Pagerank.

* Distributed Data Parallel Training: Compared the performance of different distributed data parallel training implementations in Pytorch, which includes: 
  * Sync gradient with gather and scatter
  * Sync gradient with allreduce
  * Built in Module

* Tuning_Databases_Towards_Reduced_Energy_Consumption: We addressed one of the neglected areas about auto-tuning for non-performance metrics (power consumption) while keeping the same performance. The paper was built based on a previous auto-tuning and bench-marking framework Nautilus* and made modifications to it in order to measure power consumption and performance metrics such as CPU and memory usage. Also, as part of the work, we kept tuning the five most influential knobs for different combinations to maintain high performance as found in previous research and mea-sured its power consumption.

*K. Kanellis,  R. Alagappan,  and S. Venkataraman.Too many knobs to tune?  towards faster databasetuning  by  pre-selecting  important  knobs.usenix,2020
