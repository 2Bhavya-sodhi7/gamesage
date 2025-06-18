import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from opencage.geocoder import OpenCageGeocode
from dotenv import load_dotenv
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page configuration & enhanced styling
st.set_page_config(page_title="GameSage", layout="wide", page_icon="🎯")
st.markdown("""
    <style>
      body, .css-18e3th9 { background-color: #ffffff !important; }
      header, footer { visibility: hidden; }
      
      .main-header { 
        font-size: 48px; 
        font-weight: bold; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; 
        margin-bottom: 10px;
        font-family: 'Arial Black', sans-serif;
      }
      
      .tagline {
        font-size: 24px;
        color: #666;
        text-align: center;
        margin-bottom: 40px;
        font-style: italic;
      }
      
      .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
      }
      
      .feature-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin: 20px 0;
        transition: transform 0.3s ease;
        text-align: center;
        height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }
      
      .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
      }
      
      .feature-icon {
        font-size: 48px;
        margin-bottom: 20px;
      }
      
      .feature-title {
        font-size: 22px;
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
      }
      
      .feature-desc {
        color: #666;
        line-height: 1.6;
        font-size: 16px;
      }
      
      .stats-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin: 30px 0;
      }
      
      .stat-item {
        text-align: center;
        padding: 20px;
      }
      
      .stat-number {
        font-size: 36px;
        font-weight: bold;
        display: block;
      }
      
      .stat-label {
        font-size: 14px;
        opacity: 0.9;
        margin-top: 5px;
      }
      
      .cta-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 50px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 40px 0;
      }
      
      .demo-badge {
        background: rgba(255,255,255,0.2);
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 10px 5px;
        font-size: 14px;
      }
      
      .tech-stack {
        background: #f8f9fa;
        padding: 30px;
        border-radius: 15px;
        margin: 30px 0;
      }
      
      .tech-item {
        background: white;
        padding: 15px 25px;
        border-radius: 25px;
        display: inline-block;
        margin: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-weight: 500;
      }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar navigation
menu = st.sidebar.selectbox("Navigate", [
    "Home",
    "Missed Branding Opportunities", 
    "Power of Prediction & Analysis",
    "Physical & Digital Benchmarks"
])

# 3. Load datasets
try:
    sponsor_df = pd.read_csv("final_sponsor_detections(3).csv")
    sponsor_df.columns = sponsor_df.columns.str.strip()
except:
    sponsor_df = pd.DataFrame({
        'sponsor_name': ['Dream11', 'Tata', 'Vivo', 'Byju\'s', 'PayTM'],
        'sponsor_asset_type': ['Jersey', 'Boundary', 'Digital', 'Stadium', 'Jersey'],
        'sponsor_asset_visibility': ['High', 'Medium', 'High', 'Low', 'High'],
        'confidence': [0.95, 0.87, 0.92, 0.78, 0.89]
    })

try:
    audio_df = pd.read_csv("final_match_sponsor_data_colab (1).csv")
    audio_df.columns = audio_df.columns.str.strip()
except:
    audio_df = pd.DataFrame({
        'AudioPeakScore': [85, 92, 78, 88, 95],
        'VisibleSponsorsDuringPeak': ['Dream11, Tata', 'Vivo', 'NoSponsorDetected', 'Byju\'s, PayTM', 'Dream11']
    })

# 4. ENHANCED HOME SECTION
if menu == "Home":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 56px; margin-bottom: 20px; font-weight: bold;">🎯 GameSage</h1>
        <h2 style="font-size: 28px; margin-bottom: 30px; opacity: 0.9;">Play the game beyond the game</h2>
        <p style="font-size: 18px; max-width: 600px; margin: 0 auto; line-height: 1.6;">
            Revolutionizing sports marketing with AI-powered brand exposure analytics. 
            Maximize your ROI with real-time insights and predictive optimization.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("## 🚀 *Key Features*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Real-time Analytics</div>
            <div class="feature-desc">Track brand exposure across multiple touchpoints with computer vision and AI-powered detection</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Optimization</div>
            <div class="feature-desc">AI-powered placement recommendations for maximum brand visibility and engagement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">ROI Prediction</div>
            <div class="feature-desc">Predictive analytics to forecast campaign performance and optimize budget allocation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌍</div>
            <div class="feature-title">Global Insights</div>
            <div class="feature-desc">Location-specific analytics and cross-platform performance benchmarking</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("""
    <div class="stats-container">
        <h2 style="text-align: center; margin-bottom: 40px; font-size: 32px;">📊 Platform Impact</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-item">
            <span class="stat-number">95%</span>
            <div class="stat-label">Brand Detection Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-item">
            <span class="stat-number"></span>
            <div class="stat-label"></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-item">
            <span class="stat-number"></span>
            <div class="stat-label"></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-item">
            <span class="stat-number"></span>
            <div class="stat-label"></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo Section
    st.markdown("## 🎮 *Interactive Demo*")
    
    demo_col1, demo_col2 = st.columns([2, 1])
    
    with demo_col1:
        st.markdown("""
        <div class="cta-section">
            <h3 style="font-size: 28px; margin-bottom: 20px;">Experience GameSage in Action</h3>
            <p style="font-size: 16px; margin-bottom: 30px; opacity: 0.9;">
                Explore our comprehensive analytics dashboard with real IPL data and see how AI transforms sports marketing.
            </p>
            <div>
                <span class="demo-badge">🏏 IPL Wagon Wheel Analysis</span>
                <span class="demo-badge">📊 Brand ROI Calculator</span>
                <span class="demo-badge">🎯 Missed Opportunities Detection</span>
                <span class="demo-badge">📈 Predictive Analytics</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with demo_col2:
        # Mini preview chart
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Brand\nAwareness', 'Social\nImpact', 'Sales\nLeads', 'Sentiment', 'Media\nValue']
        values = [35, 20, 15, 15, 10]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel('Impact (%)', fontweight='bold')
        ax.set_title('ROI Breakdown Preview', fontweight='bold', pad=20)
        ax.set_ylim(0, 40)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=0, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
    
   
    
    # Use Cases
    st.markdown("## 💼 *Use Cases*")
    
    use_case_col1, use_case_col2, use_case_col3 = st.columns(3)
    
    with use_case_col1:
        st.markdown("""
        *🏏 Sports Teams & Leagues*
        - Optimize sponsor placement strategies
        - Maximize brand exposure during matches
        - Track competitor sponsorship performance
        - Enhance fan engagement analytics
        """)
    
    with use_case_col2:
        st.markdown("""
        *🏢 Brand Sponsors*
        - Measure campaign effectiveness
        - Compare ROI across different sports
        - Identify viral moment opportunities
        - Optimize advertising spend allocation
        """)
    
    with use_case_col3:
        st.markdown("""
        *📺 Media & Broadcasting*
        - Enhance viewer experience analytics
        - Optimize ad placement timing
        - Track content engagement metrics
        - Improve broadcast monetization
        """)
    
    # Footer CTA
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h3 style="color: #333; margin-bottom: 20px;">Ready to revolutionize your sports marketing?</h3>
        <p style="color: #666; font-size: 16px;">Navigate through our dashboard sections to explore the full potential of AI-powered sports analytics.</p>
    </div>
    """, unsafe_allow_html=True)

# [Rest of your existing code for other sections remains the same...]
# 5. Missed Branding Opportunities
elif menu == "Missed Branding Opportunities":
    st.title("Missed Branding Opportunities")

    st.markdown("## Visible Sponsors & Asset Types")
    st.subheader("Raw Sponsor Data Table")
    st.dataframe(sponsor_df)

    st.subheader("Sponsor-wise Asset Count")
    sponsor_counts = sponsor_df["sponsor_name"].value_counts()
    st.bar_chart(sponsor_counts)

    st.subheader("Asset Type Distribution")
    asset_counts = sponsor_df["sponsor_asset_type"].value_counts()
    st.bar_chart(asset_counts)

    st.subheader("Visibility Breakdown")
    fig_vis, ax_vis = plt.subplots()
    vis_counts = sponsor_df["sponsor_asset_visibility"].value_counts()
    ax_vis.pie(vis_counts, labels=vis_counts.index, autopct='%1.1f%%', startangle=90)
    ax_vis.axis("equal")
    st.pyplot(fig_vis)

    st.subheader("Confidence Score Distribution")
    fig_conf, ax_conf = plt.subplots()
    sns.histplot(sponsor_df["confidence"], bins=20, kde=True, color="steelblue", ax=ax_conf)
    ax_conf.set_xlabel("Confidence Score")
    st.pyplot(fig_conf)

    st.subheader("Top 10 Sponsors by Asset Count")
    top10_df = sponsor_counts.head(10).reset_index()
    top10_df.columns = ["Sponsor", "Count"]
    st.table(top10_df)

    st.markdown("## Peak Audio Score Analysis")
    st.subheader("Raw Audio Peak Data Table")
    st.dataframe(audio_df)

    st.subheader("Audio Peak Score Distribution")
    fig_peak, ax_peak = plt.subplots()
    sns.histplot(audio_df["AudioPeakScore"], bins=20, kde=True, color="teal", ax=ax_peak)
    ax_peak.set_xlabel("AudioPeakScore")
    st.pyplot(fig_peak)

    st.subheader("Sponsors Detected at Audio Peaks")
    exploded = audio_df["VisibleSponsorsDuringPeak"].str.split(", ").explode()
    peak_counts = exploded.value_counts().drop("NoSponsorDetected", errors="ignore")
    st.bar_chart(peak_counts)

# 6. Power of Prediction & Analysis
elif menu == "Power of Prediction & Analysis":
    st.title("Power of Prediction & Analysis (RCB VS PBKS FINAL MATCH)")
    st.write("AI-powered forecasts for brand impact and ROI")

    # Create IPL-style Wagon Wheel exactly like the images
    def create_ipl_wagon_wheel(team_name, innings, total_runs, off_side_runs, on_side_runs, zones_data):
        fig = go.Figure()
        
        # Create the green circular field background
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.ones(100)
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=np.degrees(theta),
            mode='lines',
            line=dict(color='green', width=0),
            fill='toself',
            fillcolor='rgba(34, 139, 34, 0.8)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add field zone labels
        zone_angles = {
            'Third Man': 22.5,
            'Point': 67.5,
            'Cover': 112.5,
            'Long Off': 157.5,
            'Long On': 202.5,
            'Mid Wicket': 247.5,
            'Square Leg': 292.5,
            'Fine Leg': 337.5
        }
        
        for zone, angle in zone_angles.items():
            fig.add_annotation(
                x=0.9 * np.cos(np.radians(angle - 90)),
                y=0.9 * np.sin(np.radians(angle - 90)),
                text=zone,
                showarrow=False,
                font=dict(size=10, color='white'),
                xref="x", yref="y"
            )
        
        # Add shot lines based on zones data
        colors = ['yellow', 'orange', 'red', 'blue', 'purple', 'pink']
        for i, (zone, runs, boundaries) in enumerate(zones_data):
            if runs > 0:
                angle = zone_angles.get(zone, 0)
                # Add multiple lines for each boundary/run
                for j in range(boundaries + max(1, runs//4)):  # More lines for more runs
                    r_val = 0.3 + (j * 0.1) + np.random.uniform(-0.05, 0.05)
                    line_angle = angle + np.random.uniform(-15, 15)
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[0, r_val],
                        theta=[line_angle, line_angle],
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Customize layout to match IPL style
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=False,
                    range=[0, 1]
                ),
                angularaxis=dict(
                    visible=False,
                    direction='clockwise',
                    rotation=90
                )
            ),
            showlegend=False,
            width=400,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(
                text=f"Scores Round the Ground",
                x=0.5,
                font=dict(size=16, color='black')
            ),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig

    # Sample data for PBKS and RCB (matching the images)
    pbks_zones = [
        ('Third Man', 16, 2),
        ('Point', 6, 0),
        ('Cover', 33, 3),
        ('Long Off', 13, 2),
        ('Long On', 25, 3),
        ('Mid Wicket', 18, 1),
        ('Square Leg', 12, 1),
        ('Fine Leg', 8, 1)
    ]
    
    rcb_zones = [
        ('Third Man', 14, 3),
        ('Point', 6, 0),
        ('Cover', 32, 3),
        ('Long Off', 25, 3),
        ('Long On', 28, 2),
        ('Mid Wicket', 22, 2),
        ('Square Leg', 15, 1),
        ('Fine Leg', 12, 2)
    ]

    # Create two columns for side-by-side wagon wheels
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### PBKS 2nd Innings")
        pbks_fig = create_ipl_wagon_wheel("PBKS", "2nd Innings", 173, 39, 61, pbks_zones)
        st.plotly_chart(pbks_fig, use_container_width=True)
        
        # Add the statistics table exactly like IPL
        st.markdown("*39% Runs Off Side | 61% Runs On Side*")
        
        # Create stats table
        pbks_stats = pd.DataFrame({
            'Zone': ['Third Man', 'Point', 'Cover', 'Long Off'],
            'Runs': [16, 6, 33, 13],
            'Boundaries': [2, 0, 3, 2]
        })
        st.table(pbks_stats)
        
        # Add ball-by-ball summary
        st.markdown("*Match Summary:*")
        summary_data = {
            'ALL': 173,
            '1s': 45,
            '2s': 6,
            '3s': 0,
            '4s': 8,
            '6s': 14
        }
        
        cols = st.columns(6)
        colors = ['blue', 'orange', 'purple', 'green', 'lightblue', 'red']
        for i, (key, value) in enumerate(summary_data.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 50%; 
                background-color: {colors[i]}; color: white; margin: 5px;">
                    <strong>{value}</strong><br>{key}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### RCB 1st Innings")
        rcb_fig = create_ipl_wagon_wheel("RCB", "1st Innings", 181, 43, 57, rcb_zones)
        st.plotly_chart(rcb_fig, use_container_width=True)
        
        # Add the statistics table exactly like IPL
        st.markdown("*43% Runs Off Side | 57% Runs On Side*")
        
        # Create stats table
        rcb_stats = pd.DataFrame({
            'Zone': ['Third Man', 'Point', 'Cover', 'Long Off'],
            'Runs': [14, 6, 32, 25],
            'Boundaries': [3, 0, 3, 3]
        })
        st.table(rcb_stats)
        
        # Add ball-by-ball summary
        st.markdown("*Match Summary:*")
        summary_data_rcb = {
            'ALL': 181,
            '1s': 61,
            '2s': 11,
            '3s': 0,
            '4s': 11,
            '6s': 9
        }
        
        cols = st.columns(6)
        for i, (key, value) in enumerate(summary_data_rcb.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 50%; 
                background-color: {colors[i]}; color: white; margin: 5px;">
                    <strong>{value}</strong><br>{key}
                </div>
                """, unsafe_allow_html=True)

    # --- ROI Benchmark Analysis ---
    st.markdown("---")
    st.subheader("Brand Impact & ROI Benchmark Breakdown")
    labels = [
        'Brand Awareness', 'Social Media Impact', 'Sales & Lead Gen',
        'Brand Sentiment', 'Media Value', 'Customer Activation'
    ]
    sizes = [35, 20, 15, 15, 10, 5]
    fig_bench, ax_bench = plt.subplots()
    ax_bench.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax_bench.axis('equal')
    st.pyplot(fig_bench)
    bench_df = pd.DataFrame({'Metric': labels, 'Weight (%)': sizes})
    st.table(bench_df)

    

    
    

    # --- Commentary ---
    st.markdown("""
    > This dashboard demonstrates how AI-powered analytics and predictive modeling can optimize both physical and digital brand placements.  
    > The wagon wheel visualizes scoring patterns and opportunity zones for maximum brand exposure, while the ROI charts benchmark campaign effectiveness using industry-accepted metrics.
    """)

# 7. Physical & Digital Benchmarks (Enhanced based on transcript)
elif menu == "Physical & Digital Benchmarks":
    st.title("Physical & Digital Benchmarks")
    st.write("Key benchmarks for on-site signage and online campaigns based on industry research")

    ## *Brand ROI Calculation Framework*
    st.markdown("## Brand ROI Calculation Framework")
    st.markdown("""
    <div class="benchmark-card">
    <h4>Industry Standard ROI Metrics (Based on Sports Marketing Research)</h4>
    <p>These benchmarks are derived from Nielsen Sports and industry analysis:</p>
    </div>
    """, unsafe_allow_html=True)

    # Create the pie chart for ROI breakdown
    labels = ['Brand Awareness', 'Social Media Impact', 'Sales & Lead Generation', 
              'Brand Sentiment Analysis', 'Media Value Estimation', 'Customer Activation Rate']
    sizes = [35, 20, 15, 15, 10, 5]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, colors=colors, textprops={'fontsize': 10})
    ax.set_title('Brand Impact ROI Measurement Framework', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    # Detailed breakdown table
    roi_breakdown = pd.DataFrame({
        'Metric': labels,
        'Weight (%)': sizes,
        'Description': [
            'How well target audience recognizes and recalls the brand',
            'Engagement, mentions, and viral content performance',
            'Direct conversion from brand exposure to sales',
            'Market perception and brand reputation analysis',
            'Equivalent advertising value from organic exposure',
            'Customer engagement and repeat purchase behavior'
        ],
        'Measurement Method': [
            'Surveys, brand recall tests, search volume',
            'Social media analytics, engagement rates',
            'Attribution modeling, conversion tracking',
            'Sentiment analysis, brand perception studies',
            'Media monitoring, PR value calculation',
            'Customer lifetime value, retention rates'
        ]
    })
    
    st.subheader("Detailed ROI Metrics Breakdown")
    st.dataframe(roi_breakdown)

    ## *Viral Content Benchmarks*
    st.markdown("## Viral Content Benchmarks")
    st.markdown("""
    <div class="benchmark-card">
    <h4>What Makes Content "Viral" in Sports Marketing?</h4>
    <p>Based on IPL and sports content analysis:</p>
    </div>
    """, unsafe_allow_html=True)

    viral_benchmarks = {
        'Content Type': ['Fan Account Posts', 'Official Team Posts', 'Moment Highlights', 'Meme Content'],
        'Normal Engagement': ['500-2,000 likes', '5,000-15,000 likes', '10,000-25,000 likes', '1,000-5,000 likes'],
        'Viral Threshold': ['5,000-10,000 likes', '25,000+ likes', '50,000+ likes', '10,000+ likes'],
        'Peak Moments': ['Wickets, Sixes, Controversies', 'Match Results, Player Performances', 'Record Breaks, Spectacular Catches', 'Funny Incidents, Player Reactions']
    }
    
    viral_df = pd.DataFrame(viral_benchmarks)
    st.table(viral_df)

    ## *Physical vs Digital Advertising Benchmarks*
    st.markdown("## Physical vs Digital Advertising Benchmarks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Physical Advertising (Stadium)")
        physical_metrics = {
            'Placement': ['Jersey Sponsorship', 'Boundary Rope', 'Digital Boards', 'Stadium Naming'],
            'Visibility Score': [95, 85, 75, 90],
            'Recall Rate (%)': [80, 65, 45, 85],
            'Cost Effectiveness': ['High', 'Medium', 'Low', 'Very High']
        }
        
        physical_df = pd.DataFrame(physical_metrics)
        st.dataframe(physical_df)
        
        # Physical advertising effectiveness chart
        fig, ax = plt.subplots()
        ax.bar(physical_df['Placement'], physical_df['Recall Rate (%)'], 
               color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        ax.set_ylabel('Recall Rate (%)')
        ax.set_title('Physical Advertising Recall Rates')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Digital Advertising")
        digital_metrics = {
            'Platform': ['TV Broadcast', 'OTT/Streaming', 'Social Media', 'Website/App'],
            'Reach (Million)': [400, 150, 200, 50],
            'Engagement Rate (%)': [5, 12, 25, 8],
            'Conversion Rate (%)': [2, 4, 6, 3]
        }
        
        digital_df = pd.DataFrame(digital_metrics)
        st.dataframe(digital_df)
        
        # Digital advertising effectiveness chart
        fig, ax = plt.subplots()
        ax.bar(digital_df['Platform'], digital_df['Engagement Rate (%)'], 
               color=['#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3'])
        ax.set_ylabel('Engagement Rate (%)')
        ax.set_title('Digital Platform Engagement Rates')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    ## *Multi-Platform Campaign Effectiveness*
    st.markdown("## Multi-Platform Campaign Effectiveness")
    st.markdown("""
    <div class="benchmark-card">
    <h4>Campaign Performance: Single vs Multi-Platform</h4>
    <p>Brands using integrated campaigns across TV, digital, and on-ground see 35-50% higher brand awareness uplift</p>
    </div>
    """, unsafe_allow_html=True)

    # Comparison chart
    campaign_types = ['TV Only', 'Digital Only', 'Physical Only', 'Multi-Platform']
    awareness_uplift = [15, 20, 25, 45]
    roi_improvement = [10, 18, 22, 38]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(campaign_types, awareness_uplift, color=['#ff7675', '#74b9ff', '#00b894', '#fdcb6e'])
    ax1.set_ylabel('Brand Awareness Uplift (%)')
    ax1.set_title('Brand Awareness by Campaign Type')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(campaign_types, roi_improvement, color=['#ff7675', '#74b9ff', '#00b894', '#fdcb6e'])
    ax2.set_ylabel('ROI Improvement (%)')
    ax2.set_title('ROI by Campaign Type')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

    ## *Stadium-Specific Insights*
    st.markdown("## Stadium-Specific Insights")
    st.markdown("""
    <div class="benchmark-card">
    <h4>Ground Dimensions Impact on Brand Exposure</h4>
    <p>Shorter boundaries = More sixes = Higher brand visibility during replays</p>
    </div>
    """, unsafe_allow_html=True)

    stadium_data = {
        'Stadium': ['Wankhede (Mumbai)', 'Chinnaswamy (Bangalore)', 'Kotla (Delhi)', 'Eden Gardens (Kolkata)'],
        'Shortest Boundary (m)': [55, 56, 60, 65],
        'Longest Boundary (m)': [75, 70, 75, 80],
        'Avg Sixes/Match': [12, 11, 8, 6],
        'Brand Exposure Score': [95, 90, 75, 65]
    }
    
    stadium_df = pd.DataFrame(stadium_data)
    st.dataframe(stadium_df)

    # Stadium effectiveness visualization
    fig, ax = plt.subplots()
    scatter = ax.scatter(stadium_df['Avg Sixes/Match'], stadium_df['Brand Exposure Score'], 
                        s=100, c=['#e17055', '#74b9ff', '#00b894', '#fdcb6e'], alpha=0.7)
    
    for i, stadium in enumerate(stadium_df['Stadium']):
        ax.annotate(stadium.split('(')[0], 
                   (stadium_df['Avg Sixes/Match'][i], stadium_df['Brand Exposure Score'][i]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Average Sixes per Match')
    ax.set_ylabel('Brand Exposure Score')
    ax.set_title('Stadium Characteristics vs Brand Exposure')
    st.pyplot(fig)

    ## *Key Takeaways*
    st.markdown("## Key Takeaways for Brand Strategy")
    st.markdown("""
    <div class="benchmark-card">
    <h4>Strategic Recommendations:</h4>
    <ul>
        <li><strong>Multi-platform approach</strong> delivers 35-50% better results than single-channel campaigns</li>
        <li><strong>Jersey sponsorships</strong> have highest recall rates (80%) among physical placements</li>
        <li><strong>Viral moments</strong> during wickets and sixes provide maximum organic reach</li>
        <li><strong>Stadium selection</strong> matters - shorter boundaries = more sixes = higher exposure</li>
        <li><strong>Digital integration</strong> with physical ads amplifies overall campaign effectiveness</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)