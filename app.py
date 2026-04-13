# 1. IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime
import pytz
import gdown
from sklearn.preprocessing import MinMaxScaler
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 2. CONFIGURATIONS
st.set_page_config(page_title="Hệ thống Cảnh báo PM2.5 - KCN Đông Mai", page_icon="🏭", layout="wide")

LATITUDE = 21.0050
LONGITUDE = 106.8333
MODEL_PATH = 'ensemble_model_pm25.pkl'
GOOGLE_DRIVE_FILE_ID = '1sfpQJkLjHvj3HhRznGIpz0GfV6fo2_UV'  # ID của file trên Google Drive
RAW_DATA_PATH = 'data_dongmai_all_raw.csv'

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

BASE_MET_FEATURES = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_u', 'wind_v']


def download_model_from_drive():
    """Tự động tải mô hình từ Google Drive nếu chưa tồn tại"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Đang tải mô hình AI từ Google Drive (200MB)... Vui lòng đợi trong giây lát.'):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success("Đã tải xong mô hình!")
            except Exception as e:
                st.error(f"Lỗi khi tải mô hình từ Drive: {e}")

# 3. CORE FUNCTIONS

@st.cache_resource
def load_model_and_scaler():
    """Tải siêu mô hình AI và tái tạo bộ chuẩn hóa cho 23 biến"""

    # Tải mô hình từ Google Drive
    download_model_from_drive()

    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(RAW_DATA_PATH):
            st.error("Mô hình hoặc dữ liệu gốc chưa sẵn sàng. Vui lòng đảm bảo đã tải mô hình và dữ liệu đúng cách.")
            return None, None, None
            
        # VÁ LỖI TẠI ĐÂY: Giải nén Dictionary
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_raw['time'] = pd.to_datetime(df_raw['time'])
        df_raw['hour'] = df_raw['time'].dt.hour
        df_raw['day_of_week'] = df_raw['time'].dt.dayofweek
        df_raw['month'] = df_raw['time'].dt.month
        
        wind_dir_rad = np.radians(df_raw['wind_direction_10m'])
        df_raw['wind_u'] = -df_raw['wind_speed_10m'] * np.sin(wind_dir_rad)
        df_raw['wind_v'] = -df_raw['wind_speed_10m'] * np.cos(wind_dir_rad)
        
        # Tạo biến tích lũy (Rolling Features) cho dữ liệu gốc để fit Scaler
        for feature in BASE_MET_FEATURES:
            df_raw[f'{feature}_rolling_12h'] = df_raw[feature].rolling(window=12, min_periods=1).mean()
            df_raw[f'{feature}_rolling_24h'] = df_raw[feature].rolling(window=24, min_periods=1).mean()
            
        # Loại bỏ các dòng bị NaN do rolling
        df_raw = df_raw.dropna()
        
        scaler = MinMaxScaler()
        scaler.fit(df_raw[feature_names]) 
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Lỗi hệ thống khi tải mô hình/dữ liệu: {e}")
        return None, None, None
    

def fetch_current_environment_data():
    """Truy xuất dữ liệu Môi trường Hiện tại"""
    try:
        w_params = {"latitude": LATITUDE, "longitude": LONGITUDE, "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m", "timezone": "Asia/Bangkok"}
        w_res = requests.get(WEATHER_API_URL, params=w_params).json()['current']
        
        aq_params = {"latitude": LATITUDE, "longitude": LONGITUDE, "current": "pm2_5,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,aerosol_optical_depth", "timezone": "Asia/Bangkok"}
        aq_res = requests.get(AIR_QUALITY_API_URL, params=aq_params).json()['current']
        vn_tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(vn_tz)
        
        return {
            'pm2_5_actual': float(aq_res['pm2_5']), 
            'temperature_2m': float(w_res['temperature_2m']), 'relative_humidity_2m': float(w_res['relative_humidity_2m']),
            'precipitation': float(w_res['precipitation']), 'wind_speed_10m': float(w_res['wind_speed_10m']), 'wind_direction_10m': int(w_res['wind_direction_10m']),
            'nitrogen_dioxide': float(aq_res['nitrogen_dioxide']), 'sulphur_dioxide': float(aq_res['sulphur_dioxide']),
            'carbon_monoxide': float(aq_res['carbon_monoxide']), 'ozone': float(aq_res['ozone']), 'aerosol_optical_depth': float(aq_res['aerosol_optical_depth']),
            'hour': now.hour, 'day_of_week': now.weekday(), 'month': now.month
        }
    except Exception as e:
        st.error(f"Lỗi kết nối API: {e}")
        return None

def convert_to_aqi_vn(pm25_val):
    if pm25_val <= 25.0: return pm25_val * (50/25), "TỐT", "Không ảnh hưởng tới sức khỏe. Tự do hoạt động.", "success", "🟢"
    elif pm25_val <= 50.0: return 50 + (pm25_val - 25) * (50/25), "TRUNG BÌNH", "Nhóm nhạy cảm hạn chế ở ngoài trời lâu.", "warning", "🟡"
    elif pm25_val <= 80.0: return 100 + (pm25_val - 50) * (50/30), "KÉM", "Người bình thường bắt đầu bị ảnh hưởng. Nên đeo khẩu trang.", "warning", "🟠"
    elif pm25_val <= 150.0: return 150 + (pm25_val - 80) * (50/70), "XẤU", "Tất cả mọi người bị ảnh hưởng. Hạn chế ra ngoài.", "error", "🔴"
    else: return 200 + (pm25_val - 150), "RẤT XẤU", "Cảnh báo khẩn cấp. Mọi người ở trong nhà.", "error", "🟣"

# 4. MAIN UI EXECUTION
if 'env_data' not in st.session_state:
    vn_tz = pytz.timezone('Asia/Bangkok')
    now = datetime.now(vn_tz)
    st.session_state['env_data'] = {
        'pm2_5_actual': 20.0, 'temperature_2m': 25.0, 'relative_humidity_2m': 75.0, 'precipitation': 0.0,
        'wind_speed_10m': 5.0, 'wind_direction_10m': 180, 'nitrogen_dioxide': 15.0, 'sulphur_dioxide': 5.0, 
        'carbon_monoxide': 200.0, 'ozone': 40.0, 'aerosol_optical_depth': 0.3,
        'hour': now.hour, 'day_of_week': now.weekday(), 'month': now.month
    }

st.title("Hệ thống AI Mô phỏng & Dự báo PM2.5 KCN Đông Mai")
st.markdown("---")

col_btn, col_time = st.columns([1, 3])
with col_btn:
    if st.button("Tải Thông số Hiện tại (Từ Vệ tinh)"):
        with st.spinner('Đang kết nối API Copernicus & ECMWF...'):
            data = fetch_current_environment_data()
            if data:
                st.session_state['env_data'] = data
                st.success("Cập nhật thành công!")
with col_time:
    # [SỬA LỖI TẠI ĐÂY]
    vn_tz = pytz.timezone('Asia/Bangkok')
    current_time_vn = datetime.now(vn_tz).strftime('%H:%M - %d/%m/%Y')
    
    st.info(f"Giờ hệ thống: {current_time_vn} (Đồng bộ: Asia/Bangkok)")

base_data = st.session_state['env_data']
st.markdown(f"**Bụi mịn PM2.5 đo từ vệ tinh lúc này:** `{base_data.get('pm2_5_actual', 0):.2f} µg/m³`", unsafe_allow_html=True)

st.subheader("Tùy chỉnh Thông số (Giả lập Mô phỏng)")
col_w, col_e = st.columns(2)
with col_w:
    st.markdown("####Yếu tố Khí tượng")
    temp_in = st.number_input("Nhiệt độ (°C)", value=base_data['temperature_2m'])
    hum_in = st.number_input("Độ ẩm (%)", value=base_data['relative_humidity_2m'])
    wind_spd_in = st.number_input("Tốc độ gió (km/h)", value=base_data['wind_speed_10m'])
    wind_dir_in = st.number_input("Hướng gió (Độ)", value=base_data['wind_direction_10m'])
    precip_in = st.number_input("Lượng mưa (mm)", value=base_data['precipitation'])
with col_e:
    st.markdown("####Yếu tố Khí thải")
    no2_in = st.number_input("NO2 (µg/m³)", value=base_data['nitrogen_dioxide'])
    so2_in = st.number_input("SO2 (µg/m³)", value=base_data['sulphur_dioxide'])
    co_in = st.number_input("CO (µg/m³)", value=base_data['carbon_monoxide'])
    o3_in = st.number_input("Ozone (µg/m³)", value=base_data['ozone'])
    aod_in = st.number_input("Chỉ số quang học (AOD)", value=base_data['aerosol_optical_depth'], format="%.3f")

st.markdown("---")

model, scaler, feature_names = load_model_and_scaler()

if st.button("Chạy Mô Hình Dự đoán bụi mịn PM 2.5 hiện tại", type="primary"):
    if model is None or scaler is None:
        st.error("Lỗi: Không thể tải mô hình AI.")
    else:
        with st.spinner("AI đang tính toán phản ứng hóa học & Tích lũy 24h..."):
            wind_dir_rad = np.radians(wind_dir_in)
            wind_u_val = -wind_spd_in * np.sin(wind_dir_rad)
            wind_v_val = -wind_spd_in * np.cos(wind_dir_rad)
            
            input_dict = {
                'temperature_2m': temp_in, 'relative_humidity_2m': hum_in, 'precipitation': precip_in,
                'nitrogen_dioxide': no2_in, 'sulphur_dioxide': so2_in, 'carbon_monoxide': co_in, 
                'ozone': o3_in, 'aerosol_optical_depth': aod_in,
                'hour': base_data['hour'], 'day_of_week': base_data['day_of_week'], 'month': base_data['month'],
                'wind_u': wind_u_val, 'wind_v': wind_v_val
            }
            
            # TỰ ĐỘNG TẠO 10 BIẾN TÍCH LŨY: Giả lập thời tiết này duy trì suốt 24h qua
            for feature in BASE_MET_FEATURES:
                val = input_dict[feature]
                input_dict[f'{feature}_rolling_12h'] = val
                input_dict[f'{feature}_rolling_24h'] = val
                
            df_input = pd.DataFrame([input_dict])
            
            # Sử dụng đúng feature_names để đảm bảo thứ tự cột 100% khớp với lúc Train
            df_input = df_input[feature_names] 
            df_input[feature_names] = scaler.transform(df_input[feature_names])
            
            prediction = model.predict(df_input)[0]
            aqi_val, aqi_level, advice, state_type, icon = convert_to_aqi_vn(prediction)
            
            st.subheader("Kết Quả Dự đoán PM2.5 Hiện tại: ")
            r1, r2 = st.columns(2)
            r1.metric(label="Nồng độ PM2.5 Dự Báo", value=f"{prediction:.2f} µg/m³")
            r2.metric(label="Chỉ số VN_AQI", value=f"{aqi_val:.0f} ({aqi_level})")
            
            if state_type == "success": st.success(f"{icon} **{aqi_level}** - {advice}")
            elif state_type == "warning": st.warning(f"{icon} **{aqi_level}** - {advice}")
            else: st.error(f"{icon} **{aqi_level}** - {advice}")

st.markdown("---")

# --- BƯỚC 4: DỰ BÁO 24 GIỜ TỚI ---
st.subheader("Dự Báo PM2.5 Trong 24 Giờ Tới")
if st.button("Khởi Động Mô hình Dự báo 24h", use_container_width=True):
    if model and scaler:
        with st.spinner("Đang truy xuất dự báo khí tượng & chạy hàm Rolling 24h..."):
            w_params = {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m", "timezone": "Asia/Bangkok", "forecast_days": 2, "past_days": 1}
            aq_params = {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": "nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,aerosol_optical_depth", "timezone": "Asia/Bangkok", "forecast_days": 2, "past_days": 1}
            
            df_24h = pd.merge(pd.DataFrame(requests.get(WEATHER_API_URL, params=w_params).json()['hourly']), pd.DataFrame(requests.get(AIR_QUALITY_API_URL, params=aq_params).json()['hourly']), on='time')
            df_24h['time'] = pd.to_datetime(df_24h['time'])
            
            # Tính năng thời gian & lượng giác
            df_24h['hour'] = df_24h['time'].dt.hour
            df_24h['day_of_week'] = df_24h['time'].dt.dayofweek
            df_24h['month'] = df_24h['time'].dt.month
            wind_dir_rad = np.radians(df_24h['wind_direction_10m'])
            df_24h['wind_u'] = -df_24h['wind_speed_10m'] * np.sin(wind_dir_rad)
            df_24h['wind_v'] = -df_24h['wind_speed_10m'] * np.cos(wind_dir_rad)
            
            # TRÍCH XUẤT ĐẶC TRƯNG TÍCH LŨY TRÊN CHUỖI TƯƠNG LAI
            for feature in BASE_MET_FEATURES:
                df_24h[f'{feature}_rolling_12h'] = df_24h[feature].rolling(window=12, min_periods=1).mean()
                df_24h[f'{feature}_rolling_24h'] = df_24h[feature].rolling(window=24, min_periods=1).mean()
                
            # Lọc lấy 24h tương lai
            vn_tz = pytz.timezone('Asia/Bangkok')
            current_time = datetime.now(vn_tz)
            df_24h = df_24h[df_24h['time'] >= current_time].head(24).reset_index(drop=True)
            
            df_24h_scaled = df_24h.copy()
            df_24h_scaled = df_24h_scaled[feature_names] # Đảm bảo đúng 23 cột
            df_24h_scaled[feature_names] = scaler.transform(df_24h_scaled[feature_names])
            
            df_24h['AI_Predicted_PM25'] = model.predict(df_24h_scaled)

            # --- CHỈ CẦN THÊM ĐOẠN NÀY ĐỂ LƯU LOG ---
            import os
            os.makedirs('forecast_logs', exist_ok=True)
            # Tên file có giờ phút để không bị trùng
            log_path = f"forecast_logs/forecast_{current_time.strftime('%Y-%m-%d_%Hh%M')}.csv"
            df_24h[['time', 'AI_Predicted_PM25']].to_csv(log_path, index=False)
            # ----------------------------------------
            
            st.markdown("**1. Biểu đồ Nồng độ PM2.5 dự kiến (µg/m³)**")
            st.area_chart(df_24h.set_index('time')[['AI_Predicted_PM25']], color="#ff4b4b")

st.markdown("---")

# --- BƯỚC 5: KIỂM CHỨNG ĐỘ TIN CẬY (BACKTESTING) ---
st.subheader("Kiểm chứng Độ chính xác")

if st.button("Chạy Kiểm chứng 7 ngày qua", use_container_width=True):
    if model and scaler:
        with st.spinner("Đang xử lý hàm nhớ Rolling 24h và làm mượt biểu đồ..."):
            try:
                # 1. Tải dữ liệu 8 ngày qua (1 ngày lấy đà cho rolling, 7 ngày để hiển thị)
                w_params_past = {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m", "timezone": "Asia/Bangkok", "past_days": 8, "forecast_days": 0}
                aq_params_past = {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": "pm2_5,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,aerosol_optical_depth", "timezone": "Asia/Bangkok", "past_days": 8, "forecast_days": 0}
                
                df_past = pd.merge(pd.DataFrame(requests.get(WEATHER_API_URL, params=w_params_past).json()['hourly']), 
                                   pd.DataFrame(requests.get(AIR_QUALITY_API_URL, params=aq_params_past).json()['hourly']), 
                                   on='time').dropna()
                df_past['time'] = pd.to_datetime(df_past['time'])
                
                # Tiền xử lý gốc
                df_past['hour'] = df_past['time'].dt.hour
                df_past['day_of_week'] = df_past['time'].dt.dayofweek
                df_past['month'] = df_past['time'].dt.month
                wind_dir_rad = np.radians(df_past['wind_direction_10m'])
                df_past['wind_u'] = -df_past['wind_speed_10m'] * np.sin(wind_dir_rad)
                df_past['wind_v'] = -df_past['wind_speed_10m'] * np.cos(wind_dir_rad)
                
                # TRÍCH XUẤT ĐẶC TRƯNG TÍCH LŨY
                for feature in BASE_MET_FEATURES:
                    df_past[f'{feature}_rolling_12h'] = df_past[feature].rolling(window=12, min_periods=1).mean()
                    df_past[f'{feature}_rolling_24h'] = df_past[feature].rolling(window=24, min_periods=1).mean()
                
                # Cắt bỏ 1 ngày đầu tiên, giữ lại đúng 7 ngày
                vn_tz = pytz.timezone('Asia/Bangkok')
                seven_days_ago = datetime.now(vn_tz) - pd.Timedelta(days=7)
                df_past = df_past[df_past['time'] >= seven_days_ago].reset_index(drop=True)
                
                # Dự báo
                df_past_scaled = df_past.copy()
                df_past_scaled = df_past_scaled[feature_names]
                df_past_scaled[feature_names] = scaler.transform(df_past_scaled[feature_names])
                
                df_past['AI_Predicted_PM25'] = model.predict(df_past_scaled)
                
                # TÍNH TOÁN METRICS (Dựa trên dữ liệu gốc chưa làm mượt)
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                y_true = df_past['pm2_5'].values
                y_pred = df_past['AI_Predicted_PM25'].values
                
                mae_7days = mean_absolute_error(y_true, y_pred)
                rmse_7days = np.sqrt(mean_squared_error(y_true, y_pred))
                max_under = np.max(y_true - y_pred)
                mean_obs = np.mean(y_true)
                ioa_7days = 1 - (np.sum((y_pred - y_true)**2) / np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs))**2))
                nmb_7days = np.sum(y_pred - y_true) / np.sum(y_true)
                
                # Hiển thị Bảng Metrics
                st.markdown("#### 📐 Bảng đánh giá độ chính xác toán học chuyên sâu")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("MAE (Sai số)", f"{mae_7days:.2f} µg/m³")
                m2.metric("RMSE (Phạt lỗi lớn)", f"{rmse_7days:.2f} µg/m³")
                m3.metric("IOA (Đồng điệu)", f"{ioa_7days:.2f}")
                m4.metric("NMB (Độ lệch)", f"{nmb_7days:.4f}")
                m5.metric("Đoán hụt đỉnh", f"- {max_under:.2f} µg/m³")
                
                # ==========================================
                # KỸ THUẬT LÀM MƯỢT BIỂU ĐỒ (SMOOTHING)
                # ==========================================
                # Áp dụng hàm trung bình trượt 3 giờ để làm mượt đường cong dự báo
                # df_past['AI_Smoothed'] = df_past['AI_Predicted_PM25'].rolling(window=3, center=True, min_periods=1).mean()
                # Không áp dụng làm mượt nữa
                df_past['AI_Smoothed'] = df_past['AI_Predicted_PM25']
                
                st.markdown("#### 📈 Đối chiếu Thực tế và Dự báo")
                st.caption("Đường dự báo của AI đã được áp dụng bộ lọc làm mượt (Smoothing) để thể hiện xu hướng trực quan hơn.")
                
                # Vẽ biểu đồ Plotly tinh gọn (Đã bỏ phần chọn biến phụ)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_past['time'], y=df_past['pm2_5'], name="Thực tế (Vệ tinh)", 
                                         line=dict(color="#1f77b4", width=2.5)))
                fig.add_trace(go.Scatter(x=df_past['time'], y=df_past['AI_Smoothed'], name="AI Mô phỏng (Đã làm mượt)", 
                                         line=dict(color="#ff4b4b", width=2.5, shape='spline'))) # shape='spline' giúp bo tròn các góc
                
                fig.update_layout(
                    title_text="Đồ thị Nồng độ PM2.5", 
                    hovermode="x unified", 
                    legend=dict(orientation="h", y=-0.2, yanchor="top", x=0.5, xanchor="center"),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                fig.update_yaxes(title_text="Nồng độ PM2.5 (µg/m³)")
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Lỗi khi chạy backtesting: {e}")

# =====================================================================
# BƯỚC 6: HỆ THỐNG TỰ ĐỘNG ĐÁNH GIÁ CHÉO (QUÉT LOG)
# =====================================================================
st.markdown("---")
st.subheader("Kiểm chứng các bản Dự báo 24h đã lưu")

import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

if st.button("Quét & Kiểm tra Log Dự Báo", type="primary"):
    # Tìm tất cả file CSV, nhưng BỎ QUA các file đã có chữ '_done'
    pending_files = [f for f in glob.glob('forecast_logs/forecast_*.csv') if '_done' not in f]
    
    if not pending_files:
        st.success("Mọi thứ đã gọn gàng! Không có bản dự báo nào đang chờ (hoặc tất cả đều đã Done).")
    else:
        for file_path in pending_files:
            file_name = os.path.basename(file_path)
            st.markdown(f"**Đang xử lý:** `{file_name}`")
            
            # Đọc log AI dự báo
            df_log = pd.read_csv(file_path)
            df_log['time'] = pd.to_datetime(df_log['time'])
            
            # Lấy mốc thời gian
            end_time = df_log['time'].max()
            current_time = datetime.now(vn_tz)
            
            # LOGIC: CHƯA TỚI GIỜ THÌ BÁO VÀ BỎ QUA KHÔNG CHẠY FILE NÀY
            if current_time < end_time:
                st.warning(f"Chưa đủ 24h! Cần chờ qua thời điểm **{end_time.strftime('%H:%M ngày %d/%m')}** để vệ tinh đo được thực tế.")
                continue
                
            # LOGIC: ĐÃ QUA THỜI GIAN -> KÉO THỰC TẾ
            start_date_str = df_log['time'].min().strftime('%Y-%m-%d')
            end_date_str = end_time.strftime('%Y-%m-%d')
            
            with st.spinner(f"Đang kéo dữ liệu thực tế từ vệ tinh..."):
                try:
                    aq_params = {
                        "latitude": LATITUDE, "longitude": LONGITUDE, 
                        "hourly": "pm2_5", "timezone": "Asia/Bangkok",
                        "start_date": start_date_str, "end_date": end_date_str
                    }
                    res = requests.get(AIR_QUALITY_API_URL, params=aq_params).json()
                    df_actual = pd.DataFrame(res['hourly'])
                    df_actual['time'] = pd.to_datetime(df_actual['time'])
                    df_actual = df_actual.rename(columns={'pm2_5': 'Actual_PM25'})
                    
                    # Gộp dự báo và thực tế lại
                    df_eval = pd.merge(df_log, df_actual, on='time', how='inner')
                    df_eval_clean = df_eval.dropna() 
                    
                    if df_eval_clean.empty:
                        st.error("Dữ liệu vệ tinh tại khung giờ này chưa sẵn sàng. Thử lại sau.")
                        continue
                    
                    # Tính điểm toán học
                    mae = mean_absolute_error(df_eval_clean['Actual_PM25'], df_eval_clean['AI_Predicted_PM25'])
                    rmse = np.sqrt(mean_squared_error(df_eval_clean['Actual_PM25'], df_eval_clean['AI_Predicted_PM25']))
                    
                    # GHI ĐÈ FILE (Thêm cột thực tế) VÀ ĐỔI TÊN THÀNH _DONE
                    done_file_path = file_path.replace('.csv', '_done.csv')
                    df_eval_clean.to_csv(done_file_path, index=False)
                    os.remove(file_path) # Xóa file cũ đi
                    
                    # Hiển thị kết quả ra màn hình
                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE (Sai số)", f"{mae:.2f} µg/m³")
                    c2.metric("RMSE", f"{rmse:.2f} µg/m³")
                    c3.metric("Trạng thái", "Đã kiểm tra xong")
                    
                    # Vẽ cái biểu đồ so sánh cho VIP
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_eval_clean['time'], y=df_eval_clean['Actual_PM25'], name="Thực tế", line=dict(color="blue", width=2)))
                    fig.add_trace(go.Scatter(x=df_eval_clean['time'], y=df_eval_clean['AI_Predicted_PM25'], name="AI Dự báo", line=dict(color="red", width=2, dash='dot')))
                    fig.update_layout(title_text=f"Biểu đồ Đánh giá - {start_date_str}", hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()
                    
                except Exception as e:
                    st.error(f"Lỗi khi xử lý file: {e}")

# (BƯỚC 6: BẢNG TIÊU CHUẨN VN_AQI GIỮ NGUYÊN BÊN DƯỚI DÒNG NÀY)
st.subheader("Hệ thống Tiêu chuẩn VN_AQI (Quyết định 1459/QĐ-TCMT)")

html_table = """
<style>
    .aqi-table { width: 100%; border-collapse: collapse; text-align: left; font-family: sans-serif; }
    .aqi-table th, .aqi-table td { padding: 12px; border: 1px solid #ddd; }
    .aqi-table th { background-color: #f2f2f2; font-weight: bold; }
</style>
<table class="aqi-table">
    <tr>
        <th>VN_AQI</th>
        <th>Chất lượng không khí (Cấp độ)</th>
        <th>Ý nghĩa tác động tới sức khỏe</th>
    </tr>
    <tr style="background-color: #d4edda; color: #155724;">
        <td><b>0 - 50</b></td>
        <td><b>🟢 TỐT</b></td>
        <td>Không ảnh hưởng tới sức khỏe. Tự do hoạt động ngoài trời.</td>
    </tr>
    <tr style="background-color: #fff3cd; color: #856404;">
        <td><b>51 - 100</b></td>
        <td><b>🟡 TRUNG BÌNH</b></td>
        <td>Người bình thường không ảnh hưởng. Nhóm nhạy cảm nên hạn chế ở ngoài trời.</td>
    </tr>
    <tr style="background-color: #ffe8cc; color: #b05a00;">
        <td><b>101 - 150</b></td>
        <td><b>🟠 KÉM</b></td>
        <td>Người bình thường bị ảnh hưởng. Nhóm nhạy cảm gặp vấn đề sức khỏe. Nên đeo khẩu trang.</td>
    </tr>
    <tr style="background-color: #f8d7da; color: #721c24;">
        <td><b>151 - 200</b></td>
        <td><b>🔴 XẤU</b></td>
        <td>Tất cả bị ảnh hưởng sức khỏe, nhóm nhạy cảm bị nghiêm trọng. Hạn chế ra ngoài.</td>
    </tr>
    <tr style="background-color: #e2d9f3; color: #4a148c;">
        <td><b>201 - 300+</b></td>
        <td><b>🟣 RẤT XẤU / NGUY HẠI</b></td>
        <td>Cảnh báo khẩn cấp: Mọi người bị ảnh hưởng nghiêm trọng. Nên ở trong nhà.</td>
    </tr>
</table>
"""
st.markdown(html_table, unsafe_allow_html=True)