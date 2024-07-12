

with open('run_expr1_v1.sh', 'w') as f:
    for i in range(1, 51):
        f.write(f"python baseline_llm_expr1_v1.py {i} data_expr1_v1 &\n")
    
    
