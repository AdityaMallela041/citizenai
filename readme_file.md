# CitizenAI: Intelligent Citizen Engagement Platform
## Overview

CitizenAI is an innovative smart platform that bridges the communication gap between government and citizens through AI-powered interactions. Built upon IBM Granite's powerful foundation model, this system revolutionizes public service delivery by enabling personalized, responsive engagement at scale.

### Key Features

- **AI-Powered Conversations**: Natural language understanding with IBM Granite 3.3 2B Instruct
- **Real-time Analytics**: Sentiment analysis and citizen feedback monitoring
- **Government Portal**: Secure admin panel for officials and administrators
- **Announcement System**: Priority-based public communications
- **Secure Authentication**: Role-based access for citizens and officials
- **Responsive Design**: Accessible 24/7 through web interface

## Project Structure

```
CITIZENAI/
├── data/
│   └── sample_queries.json          # Sample citizen queries for testing
├── utils/
│   ├── __init__.py                 # Package initialization
│   ├── automation.py               # Query classification and automation
│   ├── conversation.py             # Conversation management
│   ├── sentiment.py               # Sentiment analysis utilities
│   └── visualization.py           # Data visualization components
├── app.py                         # Main Streamlit application
├── citizenai.db                   # SQLite database (auto-generated)
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM (recommended for AI model)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/citizenai.git
   cd citizenai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the platform**
   - Open your browser and navigate to `http://localhost:8501`
   - Default admin login: `username: admin`, `password: admin123`

## User Roles

### Citizens
- Ask questions about city services
- View public announcements
- Provide feedback and suggestions

### Government Officials
- Manage public announcements
- Monitor citizen sentiment
- Access analytics dashboard

### Administrators
- Approve government official accounts
- Manage user permissions
- System configuration

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **AI Framework** | IBM Granite 3.3 2B Instruct | Core conversational AI and natural language understanding |
| **Backend** | Python + Streamlit | Web application framework and user interface |
| **Conversation** | LangChain | AI service orchestration and conversation management |
| **NLP** | Transformers + NLTK | Advanced language processing and sentiment analysis |
| **Data Processing** | Pandas + NumPy | Data manipulation and analytics |
| **Visualization** | Plotly + Matplotlib | Interactive charts and dashboards |
| **Database** | SQLite | User management and data storage |
| **Authentication** | SQLAlchemy + Passlib | Secure user authentication and session management |

## Core Features

### Conversational AI
The platform uses IBM Granite 3.3 2B Instruct model to understand citizen queries in natural language and provide contextually relevant responses. The conversation manager maintains context across interactions for more meaningful exchanges.

### Sentiment Analysis
Real-time sentiment analysis tracks citizen satisfaction and identifies emerging concerns, helping government officials make data-driven decisions.

### Smart Query Classification
Automatic categorization of citizen inquiries routes questions to appropriate departments and prioritizes urgent matters.

### Analytics Dashboard
Comprehensive analytics provide insights into citizen sentiment trends, issue distribution, and key performance metrics.

### Security & Privacy
- End-to-end encryption for sensitive communications
- Role-based access control
- Secure session management
- Data protection compliance

## Database Schema

The application uses SQLite with the following main tables:

- **users**: User authentication and profile information
- **citizens**: Citizen-specific information and location data
- **government_officials**: Official credentials and department information
- **sessions**: Active user sessions and security tokens
- **announcements**: Public announcements and notifications

## Development Timeline

This project was completed in an intensive one-week hackathon sprint:

- **Day 1**: Foundation setup and IBM Granite integration
- **Day 2**: Feature development and authentication system
- **Day 3**: Administrative tools and analytics dashboard
- **Day 4**: Security implementation and data protection
- **Day 5**: Testing, refinement, and deployment

## API Integration

The platform is designed to integrate with:
- Government databases and public service portals
- Social media platforms for comprehensive feedback collection
- External APIs for enhanced functionality
- Mobile applications for broader accessibility

## Future Enhancements

- **Multilingual Support**: Serve diverse communities in multiple languages
- **Mobile Application**: Native mobile apps for iOS and Android
- **Voice Interface**: Hands-free interaction capabilities
- **IoT Integration**: Smart city device connectivity
- **Blockchain Integration**: Transparent record-keeping of interactions
- **Predictive Analytics**: Proactive service recommendations

## Contributing

We welcome contributions to improve CitizenAI. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, issues, or questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

## Acknowledgments

- IBM for providing the Granite foundation model
- Streamlit team for the excellent web framework
- Open source community for various libraries and tools
- IBM Hackathon 2025 organizers for the opportunity

---

**CitizenAI** - Bridging the gap between government and citizens through intelligent technology.