hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar \
-file /root/mapper.py \
-mapper /root/mapper.py \
-file /root/reducer.py \
-reducer /root/reducer.py \
-input ./wordcount/* \
-output ./wordcount_outdir