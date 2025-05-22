import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_sentiment_chart(sentiment_data):
    """
    Create a stacked area chart showing sentiment trends
    
    Args:
        sentiment_data: DataFrame with sentiment data
        
    Returns:
        Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=sentiment_data['date'], 
        y=sentiment_data['positive'],
        mode='lines',
        name='Positive',
        line=dict(width=0.5, color='rgb(34, 139, 34)'),
        stackgroup='one',
        groupnorm='percent'  # Normalize to 100%
    ))
    
    fig.add_trace(go.Scatter(
        x=sentiment_data['date'], 
        y=sentiment_data['neutral'],
        mode='lines',
        name='Neutral',
        line=dict(width=0.5, color='rgb(128, 128, 128)'),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=sentiment_data['date'], 
        y=sentiment_data['negative'],
        mode='lines',
        name='Negative',
        line=dict(width=0.5, color='rgb(178, 34, 34)'),
        stackgroup='one'
    ))
    
    # Update layout
    fig.update_layout(
        title='Citizen Sentiment Over Time',
        xaxis_title='Date',
        yaxis_title='Sentiment Distribution (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

def create_issue_distribution_chart(issue_data):
    """
    Create a pie chart showing distribution of citizen issues
    
    Args:
        issue_data: Dict with issue categories and counts
        
    Returns:
        Plotly figure
    """
    # Create figure
    fig = px.pie(
        values=list(issue_data.values()),
        names=list(issue_data.keys()),
        title='Citizen Issue Distribution',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    # Update layout
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_response_time_chart(response_data):
    """
    Create a bar chart showing response times by department
    
    Args:
        response_data: DataFrame with department response times
        
    Returns:
        Plotly figure
    """
    # Sort data by response time
    response_data = response_data.sort_values('avg_response_time')
    
    # Create figure
    fig = px.bar(
        response_data,
        x='department',
        y='avg_response_time',
        color='avg_response_time',
        labels={'avg_response_time': 'Average Response Time (hours)', 'department': 'Department'},
        title='Average Response Time by Department',
        color_continuous_scale='RdYlGn_r'  # Red for longer times, green for shorter
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=40, b=60),
    )
    
    return fig

def create_query_volume_chart(volume_data):
    """
    Create a line chart showing query volume over time
    
    Args:
        volume_data: DataFrame with query volumes over time
        
    Returns:
        Plotly figure
    """
    # Create figure
    fig = px.line(
        volume_data,
        x='date',
        y='query_count',
        labels={'query_count': 'Number of Queries', 'date': 'Date'},
        title='Daily Query Volume'
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

def create_satisfaction_gauge(satisfaction_score):
    """
    Create a gauge chart showing overall citizen satisfaction
    
    Args:
        satisfaction_score: Satisfaction score (0-100)
        
    Returns:
        Plotly figure
    """
    # Create figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=satisfaction_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Citizen Satisfaction"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "firebrick"},
                {'range': [50, 75], 'color': "gold"},
                {'range': [75, 100], 'color': "forestgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=250,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig