"""
Usage: python3 Pipeline/01a_convert_slim.py train|test
"""
import sys, time, json
import pyarrow as pa, pyarrow.csv as pv, pyarrow.parquet as pq

which = sys.argv[1]
SRC = {"train": "Dataset/GUIDE_Train.csv", "test": "Dataset/GUIDE_Test.csv"}[which]
OUT = f"Dataset/derived/{which}_slim.parquet"
cols = ["OrgId","IncidentId","AlertId","Timestamp","Category","IncidentGrade"]

t0 = time.time(); rows = 0; writer = None
opts = pv.ConvertOptions(include_columns=cols, column_types={c:"string" for c in ["Timestamp","Category","IncidentGrade"]})
with pv.open_csv(SRC, convert_options=opts, read_options=pv.ReadOptions(block_size=64*1024*1024)) as reader:
    for batch in reader:
        if writer is None:
            writer = pq.ParquetWriter(OUT, batch.schema, compression="zstd")
        writer.write_batch(batch)
        rows += batch.num_rows
writer.close()
m = {"source": SRC, "out": OUT, "rows": rows, "columns": cols, "elapsed_s": round(time.time()-t0,1)}
json.dump(m, open(OUT.replace(".parquet","_manifest.json"),"w"), indent=2)
print(json.dumps(m))
