

def insert_data_to_ES(data, batch_size=5):
    """批量插入数据到 Milvus"""
    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        print(f"{start_idx} - {end_idx}")

        this = [j - start_idx for j in range(start_idx, end_idx)]
        print(this)


data = [x for x in range(100, 124)]

insert_data_to_ES(data)