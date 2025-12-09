# TransStream Warehouse Workforce Optimization System

**Enterprise-grade AI-powered workforce planning and business intelligence platform**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

TransStream is a sophisticated warehouse management optimization system that combines AI-powered workforce planning with advanced business intelligence analytics. The application analyzes 1,500 m² warehouse operations (May-September 2025), optimizes employee allocation using GPT-4o, and forecasts staffing needs for October-December using linear regression.

### Business Context
- **Warehouse**: 1,500 m² logistics facility
- **Work Schedule**: 5-day week, 8-hour shifts (160 hours/month)
- **Employee Types**: 5 roles (Director, Sales, Operation Manager, Loader, Forklift Operator)
- **Operation Types**: 10 categories (manual, pallet, service operations)
- **Data Period**: Historical (May-Sept), Forecast (Oct-Dec)

## ✨ Key Features

### 🤖 AI-Powered Optimization
- **GPT-4o Integration**: Complex prompt engineering for workforce optimization
- **Formula-Based Calculations**: Mathematical formulas for each role
- **Constraint Handling**: Business rules (work hours, team size, efficiency)
- **Real-Time Analysis**: Instant optimization with detailed explanations

### 📊 Business Intelligence Dashboard
- **Executive Metrics**: 4 KPIs with month-over-month deltas
- **Interactive Charts**: 10+ Plotly visualizations
- **Comparative Analysis**: Original vs Optimized vs Forecast
- **Correlation Matrix**: Operations vs Employee relationships
- **Trend Analysis**: Multi-month operation patterns

### 📈 Forecasting Engine
- **Linear Regression**: Scikit-learn for operations prediction
- **Cascade Forecasting**: Oct → Nov → Dec sequential modeling
- **Employee Scaling**: Dynamic staffing based on operation growth/decline
- **Confidence Metrics**: Growth/decline percentages for decision support

### 🎨 Advanced UI/UX
- **Multi-Tab Interface**: Data → Optimization → Forecast → Analytics
- **Custom CSS Styling**: Centered tables, custom colors
- **Plotly Interactivity**: Hover tooltips, zoom, pan
- **Session State Management**: Persistent user choices

### 📉 Performance Optimization
- **@st.cache_data**: Model and data caching
- **Lazy Loading**: On-demand chart generation
- **Efficient Parsing**: Optimized AI response processing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Data Table  │  │ Optimization │  │  Forecast Panel │   │
│  │  Display    │  │   Controls   │  │   (Oct-Dec)     │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  OpenAI GPT-4o Optimization Engine                    │  │
│  │  - 164-line detailed prompt                           │  │
│  │  - Formula-based calculations                         │  │
│  │  - Constraint validation                              │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Linear Regression Forecasting                        │  │
│  │  - Scikit-learn LinearRegression                      │  │
│  │  - Cascade prediction (monthly sequential)           │  │
│  │  - Employee scaling logic                             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Visualization Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Plotly    │  │  Matplotlib │  │   Seaborn   │        │
│  │   Express   │  │   (backup)  │  │   (backup)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  ┌─────────────────────┐    ┌──────────────────────┐       │
│  │  df.xlsx (Source)   │───▶│  pandas DataFrame    │       │
│  │  229 data points    │    │  Cached with @cache  │       │
│  └─────────────────────┘    └──────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Data Structure

### Excel File: `df.xlsx`
**Columns (16)**:
1. **Month**: May, June, July, August, September
2-5. **Manual Operations**: Direct_Overloading_20/40, Cross_Docking_20/40
6-7. **Pallet Operations**: Pallet_Direct_Overloading, Pallet_Cross_Docking
8-11. **Service Operations**: Other_revenue, Reloading_Service, Goods_Storage, Additional_Service
12-16. **Employees**: Director, Sales, Operation_manager, Loader, Forklift_Operator

### Operation Types & Requirements

#### Manual Operations (Loader Required)
- **Direct_Overloading_20/40**: 3 hours/operation, 4 loaders/team
- **Cross_Docking_20/40**: 5 hours/operation, 4 loaders/team
- **Max parallel operations**: 3 teams simultaneously

#### Pallet Operations (Forklift_Operator Required)
- **Pallet_Direct_Overloading**: 1 hour/operation, 1 operator + 1 loader
- **Pallet_Cross_Docking**: 2 hours/operation, 1 operator + 1 loader
- **Can run parallel with manual operations**

#### Office Operations (Operation_manager Required)
- **Other_revenue**: Document processing
- **Reloading_Service**: Arrival/departure documentation
- **Goods_Storage**: Storage management
- **Additional_Service**: Extra warehouse services

## 🚀 Installation

### Prerequisites
- Python 3.9+
- OpenAI API key
- pip package manager

### Setup Steps

1. **Clone repository**
```bash
git clone <your-repo-url>
cd projekt_transstream_v3
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Verify data file**
Ensure `df.xlsx` is in project root

5. **Run application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 💻 Usage

### Step 1: View Historical Data
- Application loads with May-September warehouse data
- 16 columns: operations + employees
- Centered table with custom styling

### Step 2: Optimize Workforce
1. Click **"Employees number optimisation"** in sidebar
2. Wait for GPT-4o analysis (~10-15 seconds)
3. View optimized employee numbers
4. Read AI-generated analysis with average changes

### Step 3: Generate Forecast
1. After optimization, click **"Create forecast"**
2. System generates Oct-Nov-Dec predictions
3. View forecast table with all 16 columns
4. Analyze employee trend chart (May-Dec)

### Step 4: Executive Dashboard
1. Select operation from dropdown (e.g., "Direct_Overloading_20")
2. Choose month for analysis (May-Dec)
3. View 4 KPI metrics with deltas
4. Explore 4 interactive charts:
   - All operations overview (pie chart)
   - Operation share (donut chart)
   - Staff distribution (bar chart)
   - Month-to-month comparison (bar chart)

## 🧮 Optimization Formulas

### Loader Calculation
```
Manual_Operations = Direct_20 + Cross_20 + Direct_40 + Cross_40
Total_Hours = (Direct_20 + Direct_40) × 3h + (Cross_20 + Cross_40) × 5h
Loader_Count = MAX(2, (Total_Hours ÷ 160) × 4)
```

**Example (May)**:
- Operations: 6 + 2 + 40 + 0 = 48
- Hours: (6+40)×3 + (2+0)×5 = 148h
- Loaders: 148 ÷ 160 × 4 = 3.7 ≈ **4** (was 10 - optimized!)

### Forklift_Operator Calculation
```
Pallet_Hours = Pallet_Direct × 1h + Pallet_Cross × 2h
Operator_Count = MAX(1, Pallet_Hours ÷ 160)
```

**Example (May)**:
- Hours: 73×1 + 93×2 = 259h
- Operators: 259 ÷ 160 = 1.6 ≈ **2** (was 0 - added!)

### Operation_manager Calculation
```
Office_Operations = Other_revenue + Reloading + Storage + Additional
Manager_Count = MAX(2, MIN(5, Office_Operations ÷ 150))
```

**Example (May)**:
- Operations: 75 + 233 + 137 + 65 = 510
- Managers: 510 ÷ 150 = 3.4 ≈ **3** (was 2 - increased!)

## 📈 Forecasting Logic

### Operations Forecast (Linear Regression)
```python
from sklearn.linear_model import LinearRegression

# Map months to numbers
month_mapping = {'May': 5, 'June': 6, ..., 'December': 12}
X = [[5], [6], [7], [8], [9]]  # Training months
y = [op_counts_per_month]      # Operation counts

model = LinearRegression()
model.fit(X, y)

# Predict October
october_ops = model.predict([[10]])[0]
```

### Employee Forecast (Scaling Logic)
```python
# Based on September baseline
baseline_employees = optimized_df.iloc[-1][employee_type]
baseline_operations = sum(relevant_operations_sept)
predicted_operations = sum(relevant_operations_oct)

ops_ratio = predicted_operations / baseline_operations

# Scale with different sensitivities
if ops_ratio >= 1.2:  # 20% growth
    staff_change = 1 + (ops_ratio - 1) * sensitivity
elif ops_ratio <= 0.8:  # 20% decline
    staff_change = 1 + (ops_ratio - 1) * (sensitivity * 0.7)
else:
    staff_change = 1.0  # No change

predicted_employees = baseline_employees * staff_change
```

**Sensitivity Factors**:
- Loader: 60% growth, 40% decline
- Forklift_Operator: 70% growth, 50% decline
- Operation_manager: 40% growth, 30% decline

## 🎨 Visualization Gallery

### 1. Executive Dashboard (4 Metrics)
- **Selected Operation**: Count with % change
- **Total Operations**: Sum with % change
- **Total Staff**: Count with % change
- **Productivity**: Operations/employee with % change

### 2. All Operations Overview (Pie Chart)
- 10 operation types with percentages
- Color-coded segments
- Hover: operation name, count, percentage

### 3. Operation Share (Donut Chart)
- Selected operation vs others
- 2 segments with clear contrast

### 4. Staff Distribution (Bar Chart)
- 5 employee types
- Color-coded bars
- Text labels with counts

### 5. Month-to-Month Comparison (Bar Chart)
- Previous month vs selected month
- Green (growth) or Red (decline)
- Percentage change annotation

### 6. Employee Trend (Line Chart)
- May-December with 3 lines (Manager, Loader, Operator)
- Solid lines (historical) + Dotted lines (forecast)
- Diamond markers for predictions

### 7. Operation Trend (Line Chart)
- Selected operation across 8 months
- Blue (historical) + Orange (forecast)
- Unified hover mode

### 8. Operations by Category (Multi-Line)
- Direct Overloading
- Cross Docking
- Pallet Operations
- Service Operations
- 5-month historical trends

### 9. Correlation Matrix (Heatmap)
- 13×13 matrix (10 operations + 3 employee types)
- Color scale: red (negative) to blue (positive)
- Text annotations with correlation values

### 10. Distribution Pie Chart
- Total operations by type (May-Sept sum)
- 10 segments with names
- Sorted by size

## 🔧 Technical Details

### OpenAI Integration
```python
client = OpenAI(api_key=env["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Expert prompt..."},
        {"role": "user", "content": "164-line detailed prompt with formulas"}
    ],
    max_tokens=1000,
    temperature=0.1  # Low for deterministic results
)
```

**Prompt Engineering**:
- 164 lines of detailed instructions
- Mathematical formulas for each role
- Example calculations (May, June, July)
- Constraint definitions
- Expected output format

### Session State Management
```python
if 'show_optimization' not in st.session_state:
    st.session_state.show_optimization = False
if 'optimization_data' not in st.session_state:
    st.session_state.optimization_data = None
if 'show_forecast' not in st.session_state:
    st.session_state.show_forecast = False
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
```

**State Flow**:
1. User clicks "Optimize" → `show_optimization = True`
2. AI response stored in `optimization_data`
3. User clicks "Forecast" → `show_forecast = True`
4. Predictions cached in `forecast_data`
5. User selections persist across reruns

### Data Parsing
```python
def get_optimized_dataframe(original_df, optimized_data):
    lines = optimized_data.strip().split('\n')
    data_rows = []
    
    for line in lines:
        parts = line.split()
        if len(parts) >= 15 and parts[0] in valid_months:
            data_rows.append(parts[:16])
    
    optimized_df = pd.DataFrame(data_rows, columns=original_df.columns)
    return optimized_df
```

**Challenges Solved**:
- Variable AI output format
- Header line filtering
- Month name validation
- Column count verification

### Caching Strategy
```python
@st.cache_data
def load_data():
    df = pd.read_excel("df.xlsx")
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    return df
```

**Benefits**:
- Data loaded once per session
- Instant subsequent access
- Reduced memory usage

## 📊 Sample Data

### May 2025 (Historical)
| Operation Type | Count | Employees | Count |
|----------------|-------|-----------|-------|
| Direct_20      | 6     | Director  | 1     |
| Cross_20       | 2     | Sales     | 1     |
| Direct_40      | 40    | Manager   | 2     |
| Cross_40       | 0     | Loader    | 10    |
| Pallet_Direct  | 73    | Operator  | 0     |
| Pallet_Cross   | 93    |           |       |

**Optimization Result**:
- Loader: 10 → **4** (60% reduction)
- Operator: 0 → **2** (added)
- Manager: 2 → **3** (50% increase)

### October 2025 (Forecast)
| Operation Type | Predicted | Employees | Predicted |
|----------------|-----------|-----------|-----------|
| Direct_20      | 12        | Loader    | 4         |
| Pallet_Direct  | 95        | Operator  | 2         |
| Storage        | 180       | Manager   | 3         |

## 🎯 Use Cases

### 1. Workforce Planning
- HR departments planning monthly staffing
- Operational managers allocating resources
- Finance teams budgeting labor costs

### 2. Efficiency Analysis
- Identifying over/understaffing
- Optimizing team composition
- Improving operations per employee ratio

### 3. Capacity Forecasting
- Predicting future staffing needs
- Planning seasonal hiring
- Budgeting for growth

### 4. Performance Tracking
- Monitoring month-over-month trends
- Comparing actuals vs predictions
- Identifying operation bottlenecks

### 5. Executive Reporting
- Board presentations with visual dashboards
- Monthly KPI reviews
- Strategic planning sessions

## 🚧 Future Enhancements

### Short-term (Sprint 1-2)
- [ ] Export to PDF/Excel reports
- [ ] Email notifications for optimization results
- [ ] Historical comparison (YoY)
- [ ] User authentication

### Mid-term (Quarter 1-2)
- [ ] Multi-warehouse support
- [ ] Custom operation types
- [ ] Advanced forecasting (ARIMA, Prophet)
- [ ] What-if scenario planning

### Long-term (Year 1)
- [ ] Mobile app (React Native)
- [ ] Real-time data integration
- [ ] Machine learning recommendation engine
- [ ] Integration with ERP systems

## 📝 Code Structure

```
app.py (1369 lines)
├── Configuration (lines 1-51)
│   ├── Imports (11 libraries)
│   ├── Environment setup (.env, Streamlit secrets)
│   ├── OpenAI client initialization
│   └── Data loading (@st.cache_data)
│
├── Optimization Engine (lines 90-181)
│   ├── optimize_employees_with_ai()
│   │   ├── Data preparation (to_string)
│   │   ├── Prompt construction (164 lines)
│   │   └── GPT-4o API call
│   └── Formula implementation (embedded in prompt)
│
├── Analysis Functions (lines 183-231)
│   ├── analyze_differences() - Compare original vs optimized
│   └── Statistical calculations (mean, %, deltas)
│
├── DataFrame Processing (lines 234-277)
│   ├── get_optimized_dataframe() - Parse AI response
│   └── Robust parsing with validation
│
├── Forecasting Engine (lines 279-441)
│   ├── predict_future_operations()
│   │   ├── Linear regression (operations)
│   │   ├── Employee scaling (baseline + ratio)
│   │   └── Month-by-month cascade
│   └── Sensitivity logic for each role
│
├── Comparison Analysis (lines 444-518)
│   ├── create_comparison_analysis()
│   └── Table + chart generation
│
├── Performance Metrics (lines 520-570)
│   ├── create_performance_metrics()
│   └── Efficiency calculations
│
├── Comprehensive Charts (lines 572-627)
│   ├── create_comprehensive_charts()
│   ├── Pie chart (distribution)
│   ├── Line chart (trends)
│   └── Heatmap (correlations)
│
├── Executive Dashboard (lines 629-921)
│   ├── create_executive_dashboard()
│   │   ├── 4 KPI metrics with deltas
│   │   ├── All operations pie chart
│   │   ├── Operation share donut chart
│   │   ├── Staff distribution bar chart
│   │   └── Month comparison bar chart
│   └── Complex month/data source handling
│
├── Dependency Charts (lines 924-955)
│   ├── create_dependency_charts()
│   └── Scatter plots with trendlines
│
└── Main UI Flow (lines 957-1368)
    ├── Sidebar controls
    │   ├── Optimization button
    │   ├── Forecast button
    │   ├── Operation selector
    │   └── Month selector
    ├── Main area sections
    │   ├── Historical data table
    │   ├── Optimization results
    │   ├── Forecast table + chart
    │   ├── Operation trend analysis
    │   └── Executive dashboard
    └── Session state management
```

## 🎓 Learning Outcomes

This project demonstrates:
1. **Enterprise Architecture**: Production-grade code organization
2. **AI Integration**: Complex prompt engineering with GPT-4o
3. **Business Logic**: Formula-based optimization with constraints
4. **Data Science**: Linear regression, correlation analysis
5. **Visualization**: 10+ interactive Plotly charts
6. **UX Design**: Multi-step workflow with session state
7. **Performance**: Caching, lazy loading, efficient parsing
8. **Error Handling**: Robust data validation and fallbacks

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is for educational purposes as part of an AI/Data Science course.

## 📧 Contact

For questions or collaboration: [Your Contact Info]

---

**Built with** 🧠 using Python, Streamlit, OpenAI GPT-4o, and Plotly

**Project Stats**:
- 1,369 lines of code
- 11 dependencies
- 16 data columns
- 10 chart types
- 5 employee roles
- 10 operation types
- 3-month forecast horizon
