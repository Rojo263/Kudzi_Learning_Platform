# Kudzi Learning Platform 🎓

A comprehensive educational chatbot platform built with Streamlit, featuring AI-powered learning assistance, interactive games, and personalized study tools.

## ✨ Features

- **🤖 AI Chat Assistant** - Powered by Google Gemini AI
- **🎯 Guided Learning** - AI-powered personalized learning paths with progress tracking
- **📚 ZIMSEC Curriculum Support** - Covers all Zimbabwean education levels
- **🎮 Interactive Games** - Quiz Blitz and Flashcards for active learning
- **✍️ Personal Notes** - Create, organize, and manage study notes
- **🌐 Multilingual Support** - English and Shona language support
- **📊 Progress Tracking** - Monitor your learning journey with analytics
- **🔒 Secure Authentication** - User accounts with data privacy

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Kudzi_Chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file in the project directory
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Run the app**
   ```bash
   streamlit run kudzigemi6.py
   ```

## 🔧 Configuration

The app uses a centralized configuration system. Key settings can be modified in the `AppConfig` class:

- **API Settings**: Rate limiting, timeouts
- **Security Settings**: Input validation limits
- **Database Settings**: Database configuration
- **UI Settings**: Display preferences
- **Game Settings**: Quiz and game parameters

## 🛡️ Security Features

- **Input Validation**: All user inputs are sanitized and validated
- **SQL Injection Protection**: Parameterized queries and input filtering
- **Rate Limiting**: Prevents API abuse and ensures fair usage
- **Secure Authentication**: User session management
- **Environment Variables**: API keys stored securely

## 🗄️ Database Schema

The app uses SQLite with the following main tables:

- `users` - User accounts and authentication
- `chat_history` - Conversation history
- `user_notes` - Personal study notes
- `user_subjects` - User's selected subjects
- `user_game_scores` - Game performance tracking
- `flashcards` - Study cards
- `user_memory` - AI conversation context

## 🎯 Guided Learning Feature

The app now includes a comprehensive **AI-powered Guided Learning** system that creates personalized learning paths for each user:

### Key Features
- **🤖 AI Learning Path Generation** - Gemini AI creates customized study plans based on selected subjects and form level
- **📚 Structured Modules** - Topics organized from basic to advanced with clear learning objectives
- **🎯 Practice Exercises** - AI-generated exercises with explanations and hints
- **📊 Progress Tracking** - Real-time monitoring of completed topics and modules
- **📈 Learning Analytics** - Comprehensive dashboard showing strengths, areas for improvement, and recommendations
- **🔥 Learning Streaks** - Gamified learning with daily streak tracking
- **🔄 Adaptive Content** - AI adjusts difficulty based on user performance

### How It Works
1. Select your ZIMSEC subjects and form level
2. AI generates a personalized learning path with 5+ modules
3. Follow structured topics with clear objectives and activities
4. Complete practice exercises and mark topics as completed
5. Track progress and get AI-powered recommendations
6. Maintain learning streaks for motivation

## 🔄 Recent Improvements

### Security & Privacy
- ✅ Moved API key to environment variables
- ✅ Added input validation and sanitization
- ✅ Implemented SQL injection protection
- ✅ Added rate limiting for AI API calls

### Code Quality
- ✅ Replaced deprecated `st.experimental_rerun()` with `st.rerun()`
- ✅ Improved database connection management
- ✅ Added comprehensive error handling
- ✅ Centralized configuration management

### User Experience
- ✅ Better error messages with recovery tips
- ✅ Loading states and progress indicators
- ✅ Input validation feedback
- ✅ Rate limit notifications

### Performance
- ✅ Database connection pooling
- ✅ Efficient error handling
- ✅ Optimized API call management

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your `.env` file contains `GEMINI_API_KEY=your_key`
   - Verify the API key is valid and has sufficient quota

2. **Database Errors**
   - Check file permissions for the database file
   - Ensure SQLite is properly installed

3. **Rate Limiting**
   - The app limits AI requests to 15 per minute
   - Wait for the cooldown period if you hit the limit

### Getting Help

- Check the error messages for specific guidance
- Use the "Try Again" button for recoverable errors
- Ensure all dependencies are properly installed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for the AI capabilities
- Streamlit for the web framework
- ZIMSEC for the educational curriculum structure
- The Zimbabwean education community

---

**Made with ❤️ for Zimbabwean students** 