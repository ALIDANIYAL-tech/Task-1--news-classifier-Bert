"""
News Topic Classifier - Web Interface
Deployment: Gradio for live interaction
Model: Fine-tuned BERT
"""

import torch
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification

# Category mapping
CATEGORIES = {
    0: {"name": "World", "emoji": "🌍", "color": "#3b82f6"},
    1: {"name": "Sports", "emoji": "⚽", "color": "#10b981"},
    2: {"name": "Business", "emoji": "💼", "color": "#f59e0b"},
    3: {"name": "Sci/Tech", "emoji": "🔬", "color": "#8b5cf6"}
}

# Load model
print("Loading model...")
try:
    model = BertForSequenceClassification.from_pretrained("./news_classifier")
    tokenizer = BertTokenizer.from_pretrained("./news_classifier")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}\n")
except Exception as e:
    print(f"Error: {e}")
    print("Please run train.py first!")
    exit()

# Prediction function
def classify_headline(text):
    if not text.strip():
        return "Please enter a headline", {}
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # Get prediction
    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item()
    
    # Format result
    category = CATEGORIES[pred_class]
    result = f"""
<div style="background: linear-gradient(135deg, {category['color']}22, {category['color']}44); 
            border-left: 4px solid {category['color']}; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 10px 0;">
    <div style="font-size: 48px; text-align: center; margin-bottom: 10px;">
        {category['emoji']}
    </div>
    <h2 style="text-align: center; color: {category['color']}; margin: 10px 0;">
        {category['name']}
    </h2>
    <p style="text-align: center; font-size: 18px; color: #666;">
        Confidence: <strong>{confidence*100:.1f}%</strong>
    </p>
</div>
"""
    
    # All scores
    scores = {CATEGORIES[i]["name"]: float(probs[i]) for i in range(4)}
    
    return result, scores

# Example headlines
examples = [
    "NASA announces new Mars mission with advanced rover technology",
    "Premier League: Manchester United defeats Chelsea 3-1",
    "Stock market reaches all-time high amid economic recovery",
    "UN climate summit discusses global warming solutions",
    "Apple unveils revolutionary AI-powered smartphone",
    "Olympic champion breaks 100m world record"
]

# Custom styling
css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
#header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px 20px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}
#header h1 {
    color: white;
    margin: 0;
    font-size: 2.5em;
    font-weight: 700;
}
#header p {
    color: rgba(255,255,255,0.9);
    margin-top: 10px;
    font-size: 1.2em;
}
.info-box {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 20px 0;
}
"""

# Build interface
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="purple")) as demo:
    
    # Header
    gr.HTML("""
        <div id="header">
            <h1>🤖 News Topic Classifier</h1>
            <p>AI-Powered Headline Categorization using BERT</p>
        </div>
    """)
    
    gr.Markdown("""
    ### Enter a news headline to classify it into one of four categories:
    **🌍 World** | **⚽ Sports** | **💼 Business** | **🔬 Science & Technology**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="News Headline",
                placeholder="Type or paste a news headline here...",
                lines=4
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="lg")
        
        with gr.Column(scale=1):
            output_html = gr.HTML(label="Prediction")
            output_label = gr.Label(label="Confidence Scores", num_top_classes=4)
    
    # Examples
    gr.Markdown("### 💡 Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=[output_html, output_label],
        fn=classify_headline,
        cache_examples=False
    )
    
    # Info section
    gr.HTML("""
        <div class="info-box">
            <h3>📊 Model Information</h3>
            <ul style="line-height: 1.8;">
                <li><strong>Model:</strong> BERT (bert-base-uncased)</li>
                <li><strong>Dataset:</strong> AG News (120,000 articles)</li>
                <li><strong>Accuracy:</strong> ~93%</li>
                <li><strong>Categories:</strong> 4 (World, Sports, Business, Sci/Tech)</li>
                <li><strong>Framework:</strong> Transformers + PyTorch</li>
            </ul>
        </div>
    """)
    
    # Button actions
    submit_btn.click(
        fn=classify_headline,
        inputs=input_text,
        outputs=[output_html, output_label]
    )
    
    input_text.submit(
        fn=classify_headline,
        inputs=input_text,
        outputs=[output_html, output_label]
    )
    
    clear_btn.click(
        fn=lambda: ("", None, None),
        outputs=[input_text, output_html, output_label]
    )

# Launch
if __name__ == "__main__":
    print("Starting News Classifier...")
    print("Open browser at: http://127.0.0.1:7860\n")
    demo.launch(server_port=7860, share=True)