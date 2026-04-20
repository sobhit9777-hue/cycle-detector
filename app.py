from flask import Flask, render_template, request, jsonify
from cycle_engine import CycleDetector
import json
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

app = Flask(__name__)

# Global detector instance
detector = CycleDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_point', methods=['POST'])
def add_point():
    data = request.json
    try:
        detector.add_manual_point(
            data['datetime'],
            data['price'],
            data['type']
        )
        return jsonify({
            'status': 'ok',
            'points': [{
                'datetime': p['datetime'],
                'price': p['price'],
                'type': p['type']
            } for p in detector.manual_points]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/clear_points', methods=['POST'])
def clear_points():
    detector.clear_points()
    return jsonify({'status': 'ok'})

@app.route('/set_timeframe', methods=['POST'])
def set_timeframe():
    data = request.json
    detector.set_timeframe(data['timeframe'])
    return jsonify({'status': 'ok', 'timeframe': detector.timeframe})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    
    # Set timeframe
    detector.set_timeframe(data.get('timeframe', '5m'))
    
    bars_ahead = int(data.get('bars_ahead', 50))
    num_cycles = int(data.get('num_cycles', 5))
    
    # Run analysis
    result = detector.full_analysis(bars_ahead, num_cycles)
    
    if isinstance(result, dict) and 'error' in result:
        return jsonify(result)
    
    # Create Plotly charts
    charts = create_charts(result)
    
    return jsonify({
        'status': 'ok',
        'result': {
            'cycles': result['cycles'],
            'future': result['future'],
            'confluence': result['confluence'],
            'dominant_cycle': result['dominant_cycle'],
            'total_cycle_strength': result['total_cycle_strength'],
            'total_bars': result['total_bars'],
            'num_points': result['num_points']
        },
        'charts': charts
    })

def create_charts(result):
    """Create Plotly charts for visualization"""
    
    charts = {}
    
    # ========================
    # CHART 1: Price + Cycles + Future
    # ========================
    fig1 = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["🔄 Price + Cycle Reconstruction + Prediction", 
                        "📊 Cycle Spectrum"],
        vertical_spacing=0.12
    )
    
    time_arr = result['time_array']
    original = result['original_price']
    reconstructed = result['reconstructed_price']
    future = result['future']
    
    # Original price
    if len(original) > 0:
        fig1.add_trace(go.Scatter(
            x=time_arr, y=original,
            name="Original Price",
            line=dict(color='white', width=1),
            opacity=0.6
        ), row=1, col=1)
    
    # Reconstructed (filtered)
    if len(reconstructed) > 0:
        fig1.add_trace(go.Scatter(
            x=time_arr, y=reconstructed,
            name="Cycle Reconstruction",
            line=dict(color='#00FFFF', width=3)
        ), row=1, col=1)
    
    # Manual points
    for p in result['manual_points']:
        # Find bar index
        idx = None
        for i, t in enumerate(time_arr):
            mp_minutes = 0
            try:
                from datetime import datetime
                mp_dt = datetime.strptime(p['datetime'], "%Y-%m-%d %H:%M")
                first_dt = datetime.strptime(result['manual_points'][0]['datetime'], "%Y-%m-%d %H:%M")
                mp_minutes = (mp_dt - first_dt).total_seconds() / 60
                mp_bar = mp_minutes / detector.tf_minutes
                if abs(t - mp_bar) < 2:
                    idx = i
                    break
            except:
                pass
        
        color = '#00FF00' if p['type'] == 'bottom' else '#FF0000'
        symbol = 'triangle-up' if p['type'] == 'bottom' else 'triangle-down'
        
        fig1.add_trace(go.Scatter(
            x=[idx if idx else 0], y=[p['price']],
            mode='markers+text',
            name=f"{p['type'].upper()} {p['price']}",
            marker=dict(size=15, color=color, symbol=symbol),
            text=[f"{p['type'][0].upper()}\n{p['price']}"],
            textposition='top center',
            textfont=dict(color=color, size=10)
        ), row=1, col=1)
    
    # Future prediction
    if future and len(future.get('future_price', [])) > 0:
        fig1.add_trace(go.Scatter(
            x=future['future_time'],
            y=future['future_price'],
            name="🔮 Future Prediction",
            line=dict(color='#FFD700', width=2, dash='dot')
        ), row=1, col=1)
        
        # Future tops
        for ft in future.get('future_tops', []):
            fig1.add_trace(go.Scatter(
                x=[ft['bar']], y=[ft['price']],
                mode='markers',
                name=f"Future Top {ft['price']}",
                marker=dict(size=12, color='red', symbol='triangle-down'),
                showlegend=False
            ), row=1, col=1)
        
        # Future bottoms
        for fb in future.get('future_bottoms', []):
            fig1.add_trace(go.Scatter(
                x=[fb['bar']], y=[fb['price']],
                mode='markers',
                name=f"Future Bottom {fb['price']}",
                marker=dict(size=12, color='lime', symbol='triangle-up'),
                showlegend=False
            ), row=1, col=1)
    
    # Cycle Spectrum (bar chart)
    cycles = result['cycles']
    if len(cycles) > 0:
        periods = [c['period_time'] for c in cycles[:10]]
        strengths = [c['strength_pct'] for c in cycles[:10]]
        colors = ['#FF6B6B' if i == 0 else '#4ECDC4' if i < 3 else '#95E1D3' 
                  for i in range(len(periods))]
        
        fig1.add_trace(go.Bar(
            x=periods, y=strengths,
            name="Cycle Strength",
            marker_color=colors,
            text=[f"{s}%" for s in strengths],
            textposition='outside'
        ), row=2, col=1)
    
    fig1.update_layout(
        template='plotly_dark',
        height=800,
        title="🔄 Euler-Fourier Cycle Analysis",
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)'),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e'
    )
    
    fig1.update_xaxes(title_text="Bar Index", row=1, col=1, gridcolor='#2a2a4a')
    fig1.update_yaxes(title_text="Price", row=1, col=1, gridcolor='#2a2a4a')
    fig1.update_xaxes(title_text="Cycle Period", row=2, col=1, gridcolor='#2a2a4a')
    fig1.update_yaxes(title_text="Strength %", row=2, col=1, gridcolor='#2a2a4a')
    
    charts['main'] = json.loads(plotly.io.to_json(fig1))
    
    # ========================
    # CHART 2: Individual Cycles
    # ========================
    if len(cycles) > 0 and len(time_arr) > 0:
        num_show = min(5, len(cycles))
        fig2 = make_subplots(
            rows=num_show, cols=1,
            subplot_titles=[f"Cycle #{i+1}: Period={cycles[i]['period_time']} | "
                           f"Strength={cycles[i]['strength_pct']}%" 
                           for i in range(num_show)],
            vertical_spacing=0.05
        )
        
        colors_list = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6BCB77', '#FF8E71']
        N = len(time_arr)
        
        for i in range(num_show):
            c = cycles[i]
            # Generate individual cycle wave
            cycle_wave = [c['amplitude'] * np.cos(2 * np.pi * c['frequency'] * t + c['phase']) 
                         for t in range(N)]
            
            fig2.add_trace(go.Scatter(
                x=list(range(N)),
                y=cycle_wave,
                name=f"Cycle {c['period_time']}",
                line=dict(color=colors_list[i % len(colors_list)], width=2),
                fill='tozeroy',
                fillcolor=f"rgba({int(colors_list[i % len(colors_list)][1:3], 16)},"
                         f"{int(colors_list[i % len(colors_list)][3:5], 16)},"
                         f"{int(colors_list[i % len(colors_list)][5:7], 16)},0.1)"
            ), row=i+1, col=1)
            
            # Zero line
            fig2.add_hline(y=0, line_dash="dash", line_color="gray", 
                          opacity=0.3, row=i+1, col=1)
        
        fig2.update_layout(
            template='plotly_dark',
            height=200 * num_show + 100,
            title="🌊 Individual Cycle Waves (Euler Components)",
            showlegend=True,
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e'
        )
        
        charts['cycles'] = json.loads(plotly.io.to_json(fig2))
    
    # ========================
    # CHART 3: Confluence Heatmap
    # ========================
    confluence = result.get('confluence', [])
    if len(confluence) > 0 and len(time_arr) > 0:
        fig3 = go.Figure()
        
        # Price line
        fig3.add_trace(go.Scatter(
            x=time_arr, y=original,
            name="Price",
            line=dict(color='white', width=1)
        ))
        
        # Confluence points
        for cp in confluence:
            color = '#FF0000' if cp['type'] == 'top' else '#00FF00'
            fig3.add_trace(go.Scatter(
                x=[cp['bar']], y=[cp['price_estimate']],
                mode='markers+text',
                name=f"Confluence {cp['type']}",
                marker=dict(size=20, color=color, symbol='star',
                           line=dict(width=2, color='white')),
                text=[f"⭐ {cp['type'].upper()}\nScore: {cp['score']}"],
                textposition='top center',
                textfont=dict(color=color),
                showlegend=False
            ))
        
        fig3.update_layout(
            template='plotly_dark',
            height=400,
            title="⭐ Cycle Confluence Points (High Probability Turns)",
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e'
        )
        
        charts['confluence'] = json.loads(plotly.io.to_json(fig3))
    
    return charts

if __name__ == '__main__':
    app.run(debug=True, port=5000)
