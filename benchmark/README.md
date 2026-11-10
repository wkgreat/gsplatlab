# 执行方法

```
benchmark.py \ 
    --repleat 10 \
    --report bnechmark-report.md \
    python3 pygsplat/test/test_cugsplat.py --data data/Ceramic_Studio.ply --scene data/Ceramic_Studio.scene.json
```

如果对应的report没有，则创建；如果有则追加 

# 输出报告
benchmark/report.md
每次测试往里面追加，字段含义
| 字段 | 含义 |
|---|---|
| datetime | 测试日期时间点(yyyyMMdd-HHmmss) |
| gpu | gpu型号+gpu显存大小 |
| cuda | cuda版本 |
| script | 使用的脚本名称 |
| data | 使用的数据文件名称 |
| scene | 使用的场景文件名称 |
| repeat | 重复执行的次数 |
| total time(ms) | 总时长 |
| mean time (ms) | 平均时长 |
| median time (ms) | 中位时长 |
| min time(ms) | 最短时长 |
| max time(ms) | 最大时长 |
| 90 line | 90线 |
| 99 line | 99线 |