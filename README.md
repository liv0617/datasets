# datasets

**This is a repo for datasets and loaders bundled into a package personal projects and is not intended to be used by others. Actual data files may be missing due to being very large.**

##### Human Genome Dataset

- generated by getting 512 length sequences beginning at every position
- used [Genome assembly GRCh38](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/) as data source
- uses custom tokenizer:
    - 4096 length vocab generated via a BPE-like algorithm that takes a random sample of sequences and finds most frequent token pair and replaces it with a new token until vocab length is reached (base tokens were just 'A', 'C', 'G', 'T'). 
- excluded sequences containing 'N's but probably better to replace with the unknown token in the tokenizer.
- typically training until a fixed number of tokens trained on due to extensive size.