import os
import sqlite3
import pandas as pd
import numpy as np

def init_telco_database(db_path="database/telco_ops.db"):
    """Tạo database SQLite và chèn dữ liệu mẫu KPI trạm, lịch sử cảnh báo."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Tạo bảng cell_kpi (Thông số kỹ thuật trạm vô tuyến)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cell_kpi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        cell_id TEXT,
        site_name TEXT,
        rsrp_dbm REAL,
        sinr_db REAL,
        prb_utilization_pct REAL,
        throughput_dl_mbps REAL,
        call_drop_rate_pct REAL
    )
    ''')
    
    # 2. Tạo bảng alarm_logs (Lịch sử cảnh báo thiết bị/trạm)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alarm_logs (
        alarm_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        cell_id TEXT,
        site_name TEXT,
        severity TEXT,
        alarm_description TEXT,
        status TEXT
    )
    ''')
    
    # Tạo dữ liệu giả lập (Mock Data) cho 7 ngày gần nhất
    np.random.seed(42)
    n_records = 150
    
    timestamps = pd.date_range(start="2026-09-01 00:00:00", periods=n_records, freq="H").strftime('%Y-%m-%d %H:%M:%S')
    cell_ids = [f"VNPT_BTS_{i:03d}" for i in np.random.randint(1, 15, n_records)]
    site_names = [f"Trạm Cần Giờ {i}" if i < 5 else f"Trạm Quận 7 {i}" for i in [int(c.split('_')[-1]) for c in cell_ids]]
    
    rsrp = np.random.normal(-88, 8, n_records).clip(-115, -70)
    sinr = np.random.normal(12, 4, n_records).clip(-2, 25)
    prb = np.random.normal(65, 15, n_records).clip(10, 98)
    dl = np.random.normal(45, 12, n_records).clip(5, 120)
    cdr = np.random.normal(0.8, 0.3, n_records).clip(0.1, 5.0)
    
    # Chèn dữ liệu vào bảng cell_kpi
    for i in range(n_records):
        cursor.execute('''
        INSERT INTO cell_kpi (timestamp, cell_id, site_name, rsrp_dbm, sinr_db, prb_utilization_pct, throughput_dl_mbps, call_drop_rate_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamps[i], cell_ids[i], site_names[i], float(rsrp[i]), float(sinr[i]), float(prb[i]), float(dl[i]), float(cdr[i])))
        
    # Chèn dữ liệu giả lập cảnh báo (Alarms)
    severities = ['CRITICAL', 'MAJOR', 'MINOR']
    descriptions = [
        'Mất nguồn điện lưới (AC Failure)', 
        'Nghẽn tài nguyên khối truyền dẫn PRB cao > 90%', 
        'Lỗi suy hao tín hiệu thu RSRP bất thường', 
        'Mất kết nối giao diện S1/X2'
    ]
    
    for i in range(20):
        cursor.execute('''
        INSERT INTO alarm_logs (timestamp, cell_id, site_name, severity, alarm_description, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            timestamps[i*7], 
            cell_ids[i*7], 
            site_names[i*7], 
            np.random.choice(severities, p=[0.2, 0.5, 0.3]), 
            np.random.choice(descriptions),
            np.random.choice(['ACTIVE', 'RESOLVED'])
        ))
        
    conn.commit()
    conn.close()
    print(f"[Database Init] Đã khởi tạo thành công cơ sở dữ liệu tại: {db_path}")

if __name__ == "__main__":
    init_telco_database()