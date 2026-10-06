# pairwise-crdt

只依赖 Python 标准库的内存 CRDT 合并内核：节点 id 与写入时间戳都由调用方注入，
内核不读真实时钟、不取随机数，也不做任何 I/O。

- `crdt/core.py` — 合并内核：向量时钟、LWW 寄存器、observed-remove 集合与副本合并。
- `tests/test_core.py` — 验收用例。

## 运行测试

在项目根目录执行：

```
python3 -m unittest discover -s tests -v
```

Windows 上如果没有 `python3`，可用：

```
python -m unittest discover -s tests -v
```
