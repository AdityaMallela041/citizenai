
# CitizenAI – Intelligent Citizen Engagement Platform

**CitizenAI** is an AI-driven civic engagement platform developed with Streamlit and IBM Granite. It enables seamless interaction between citizens and local government authorities through features like complaint management, AI-based assistance, document summarization, and real-time analytics.

---

## Features

- AI Assistant powered by IBM Granite for handling citizen queries.
- Smart Complaint System for citizens to submit and track complaints; government officials can assign, manage, and resolve them.
- Analytics Dashboard for visualizing public sentiment, issue distribution, and system health.
- Document Processor for uploading PDF, DOCX, or TXT files and automatically extracting summaries and key points.
- Secure login and role-based access for citizens, government officials, and administrators.
- Government registration approval panel for admin control.
- Announcement system for broadcasting official updates and alerts.

---

## Technology Stack

| Category         | Tools and Frameworks                       |
|------------------|---------------------------------------------|
| Frontend         | Streamlit, HTML/CSS                         |
| AI Models        | IBM Granite 3.3 (via HuggingFace Transformers) |
| Visualization    | Plotly, Pandas                              |
| Natural Language | NLTK, Transformers, LangChain              |
| Backend/Utils    | Python, SQLite, uuid, hashlib, dotenv       |

---

## Folder Structure

```
citizenai/
│
├── app.py                     # Main application entry point
├── assets/                   # Static assets (e.g., background image)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── utils/                    # Modular backend code
│   ├── automation.py
│   ├── complaint.py
│   ├── conversation.py
│   ├── document_processor.py
│   ├── resource_allocation.py
│   ├── resource_visualization.py
│   ├── sentiment.py
│   ├── visualization.py
```

---

## Supported Document Types

The system supports the following file formats for document analysis:

- PDF (`.pdf`)
- Word Documents (`.docx`)
- Plain Text (`.txt`)

Uploaded files are processed using AI to extract summaries, key points, and enable question answering.

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AdityaMallela041/citizenai.git
   cd citizenai
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

---

## Maintainers

- Aditya Mallela – Lead Developer
- Akshith Jalagari – UI/UX Contributor
- Pavan Kumar Mudumba – Data & Analytics Contributor

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
