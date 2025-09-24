# CS336 课程设计 Assignment 1

## 1. 实现一个BPE Tokenizer

### 原理

每次统计最高频率的`byte pair`，将其合并为一个新的token，重复这个过程直到达到预设的词汇表大小。

### 优化思路

每次都扫描全部文本，合并并统计太消耗性能，可以统计每个token的邻居，减少扫描次数。

### 步骤

- 初始化词库并进行预处理，首先将256个字符转为bytes并导入vocab，将special_tokens进行encode并导入vocab
- 编译pre_tokenize和查找special_tokens的正则表达式
- 准备保存以下数据
  - pre_token的频数和划分方式，pre_token_dict
  - bytes对的频数以及位于哪些token中，pair_dict，用于在merge过程中查询并进行更新
  - merges
- 首先使用find_chunk_boundaries对文本进行预处理，将文本划分为数个chunk，注意需要将chunk解码为str才能应用正则表达式
- 分别处理每个chunk
  - 注意要对chunk中的换行符进行替换处理`chunk = re.sub(r"\r\n?", "\n", chunk)`
  - ![image-20250922184450525](./assets/image-20250922184450525.png)
  - 使用special_tokens的正则表达式将chunk划分为数个part，形成一个parts列表，如果没有特别token，就只有一个part
  - 对于每个part
    - 使用pre_tokenize正则表达式将part划分为数个pre_token，使用迭代器每次读入一个token进行处理
    - 对于每个pre_token
      - 首先编码为bytes
      - 在pre_token_dict中更新值域：pre_token_dict[][0] = 频数, pre_token_dict[][1] = 划分方式，也就是按照bytes划分的列表
      - 遍历bytes划分方式，更新pair_dict中每个bytes对的值域：pair_dict[][0] = 频数, pair_dict[][1] = 出现该bytes对的token列表
      - 注意这里有个小区别：pre_token_dict中存储的是bytes列表，而pair_dict中存储的是一个set,保存该pair所在的token，用于在后续合并时更新计数
      - 注意通过trycatch捕获键错误
- 完成初始化统计后，开始尝试合并，统计len(vocab)直到大小达到指定数量
- 开始循环merge
  - 寻找最大频率的bytes对
  - 找到这些bytes对中字典序最大的bytes对，将该bytes对添加到merges列表中
  - 将该bytes对合并为一个新的token，加入vocab
  - 根据pair_dict所存储的token列表，找到该bytes对所在的tokens，进行更新，注意这里的数据类型是set，需要转为list进行遍历
    - 在token_dict中找到每一个token，得到其频数tok_freq以及划分方式
    - 统计旧划分方式的相邻bytes对，根据merge的bytes对，统计新划分方法下的相邻bytes对
    - 更新pre_token_dict中该token的划分方式
    - 统计两种划分方式下的bytes对，进行计数，在pair_dict中进行更新
      - pair_dict的频数部分减去旧划分方式下的count * tok_freq，并在set部分中删除该token
      - pair_dict的频数部分加上新划分方式下的count * tok_freq，并在set部分中添加该token
      - 注意通过trycatch捕获键错误
      

## 2. 使用BPE Tokenizer对文本进行Encode和Decode

### 原理

- 按照实验要求实现一个BPETokenizer类，能够读入vocab，merge，special_tokens
- 注意special_token存在一个是另一个的子串的情况，所以需要将special_token按长度从大到小进行排序并编译为正则表达式，优先匹配更长的special_token，防止被覆盖
- 主要是encode过程，同样将文本切分为token并编码为bytes后进行单字符划分，每次遍历相邻的两个bytes_pair，找到其中在merge列表中index最小的，将这两个合并（与训练过程中合并完全相同）形成新的划分，之后多次遍历并合并直到不存在可以合并的bytes为止

### 优化

- 暂时没有想到

### 步骤

- 编译pre_tokenize和special_token的正则表达式，注意special_token需要按长度排序
- ![image-20250923140447246](./assets/image-20250923140447246.png)
- 首先根据special_token将文本分段，在将其中的特殊token提取出来
  - 分段的部分再次进行tokenize，之后依次将所有tokenize的部分和特殊token加入列表中
- 遍历列表，准备进行编码
  - 如果是特殊token，就先不做处理
  - 如果是正常分词token
    - 先单字符划分
    - 遍历所有adj_pair的组合，找到其中在merge列表中index最小的pair，将这个pair合并，更新划分方式（pair合并后会成为一个bytes）

## 3. 实现LanguageModelArchitecture

### 原理

- 依次实现线性层、Embedding层
- 初始化权重时，注意使用`self.weight = nn.init.trunc_normal_()`
- 注意这里没有偏置项