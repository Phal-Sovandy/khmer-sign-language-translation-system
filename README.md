# Khmer Sign Language Translation System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-ISC-green.svg)
![React](https://img.shields.io/badge/React-19.2.0-61DAFB?logo=react)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)

**A web-based application that translates Khmer Sign Language gestures into readable text and speech using computer vision and machine learning.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## Overview

The Khmer Sign Language Translation System is a web-based application developed as a Capstone Project for Year 3 students in the Data Science specialization and Computer Science department. This system assists communication by translating Khmer sign language gestures into readable text and spoken output through a simple, accessible web interface.

### Key Highlights

- **Real-time Gesture Recognition**: Uses MediaPipe for hand landmark detection and PyTorch-based ML models for gesture classification
- **120+ Supported Signs**: Currently recognizes over 120 Khmer sign language gestures with continuous expansion
- **Text-to-Speech**: Converts detected signs into both text and audio output
- **No Installation Required**: Runs entirely in modern web browsers
- **Accessible Design**: Built with accessibility and user experience in mind

## Features

### Core Functionality

- **Real-time Video Processing**: Live camera feed with hand landmark visualization
- **AI-Powered Recognition**: Deep learning model fine-tuned on Khmer sign language dataset
- **Text Translation**: Instant conversion of gestures to readable text
- **Speech Output**: Text-to-speech conversion with voice gender selection
- **Customizable Settings**: Adjustable sensitivity, buffer size, and camera preferences

### User Experience

- **Modern UI/UX**: Beautiful, responsive interface built with React and Tailwind CSS
- **Keyboard Shortcuts**: Quick camera toggle with 'S' key
- **Responsive Design**: Works on desktop, laptop, and tablet devices
- **Multi-page Application**: Homepage, About, Documentation, Contact, and Demo pages
- **Interactive Demo**: Try the system directly from the homepage

### Technical Features

- **Fast Performance**: Optimized with Vite for lightning-fast development and builds
- **Lazy Loading**: Code splitting for improved load times
- **MediaPipe Integration**: Real-time hand tracking and landmark visualization
- **RESTful API**: Flask backend with CORS support for seamless frontend-backend communication
- **Modular Architecture**: Well-organized codebase with clear separation of concerns

## Tech Stack

### Frontend

- **React 19.2.0** - UI library
- **React Router DOM 7.9.6** - Client-side routing
- **Vite 7.2.4** - Build tool and dev server
- **Tailwind CSS 4.1.17** - Utility-first CSS framework
- **GSAP 3.13.0** - Animation library
- **MediaPipe Hands** - Hand landmark detection
- **EmailJS** - Contact form integration

### Backend

- **Flask 3.0.0** - Python web framework
- **PyTorch 2.1.0** - Deep learning framework
- **OpenCV 4.8.1.78** - Computer vision library
- **MediaPipe 0.10.8** - Hand tracking
- **NumPy 1.24.3** - Numerical computing
- **Flask-CORS 4.0.0** - Cross-origin resource sharing

## Installation

### Prerequisites

- **Node.js** 18.0.0 or higher
- **npm** 9.0.0 or higher
- **Python** 3.11 or higher
- **pip** (Python package manager)
- Modern web browser with camera access

### Frontend Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Phal-Sovandy/Khmer-Sign-Language-Translation-System.git
   cd Khmer-Sign-Language-Translation-System
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Set up environment variables** (optional)

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your EmailJS credentials if you want to use the contact form.

4. **Start the development server**

   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

### Backend Setup

For detailed backend setup instructions, see [backend/README.md](./backend/README.md).

**Quick Start:**

1. **Navigate to backend directory**

   ```bash
   cd backend
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**

   ```bash
   python server.py
   ```

   The backend API will be available at `http://localhost:3000`

## Usage

### Running the Application

1. **Start the backend server** (required for gesture recognition)

   ```bash
   cd backend
   python server.py
   ```

2. **Start the frontend development server** (in a separate terminal)

   ```bash
   npm run dev
   ```

3. **Open your browser**

   Navigate to `http://localhost:5173` and allow camera access when prompted.

### Using the Demo

1. **Navigate to Demo Page**: Click "Try Demo" on the homepage or go to `/demo`
2. **Enable Camera**: Click the camera button or press 'S' to toggle the camera
3. **Perform Signs**: Position your hands in front of the camera and perform Khmer sign language gestures
4. **View Results**: Detected text will appear in the text area below the video feed
5. **Listen**: Click the speaker button to hear the text-to-speech output
6. **Adjust Settings**: Click the settings icon to customize sensitivity, voice, and other options

### Keyboard Shortcuts

- **S** - Toggle camera on/off
- **ESC** - Close modals

### API Usage

The backend provides a RESTful API for gesture prediction:

**Endpoint:** `POST /predict_image`

**Request Body:**

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**

```json
{
  "label": "sign_name",
  "confidence": 95.5,
  "landmarks": [...]
}
```

For more details, see [backend/README.md](./backend/README.md).

## Project Structure

```
Sign Language Translation System - Web/
├── backend/                      # Python Flask Backend
│   ├── model/                   # Trained ML model files
│   ├── templates/               # HTML templates
│   ├── server.py                # Flask server application
│   ├── requirements.txt         # Python dependencies
│   └── README.md                # Backend documentation
│
├── src/                         # React Frontend Application
│   ├── components/              # Reusable React components
│   │   ├── demo/                # Demo page components
│   │   ├── layout/              # Layout components (Header, Footer)
│   │   ├── ui/                  # UI components (Tooltip, MediaWrapper)
│   │   └── visuals/             # Visual/decoration components
│   ├── pages/                   # Page components
│   │   └── demo/                # Demo page
│   ├── sections/                # Section components
│   │   └── demo/                # Homepage demo section
│   ├── hooks/                   # Custom React hooks
│   ├── config/                  # Configuration files
│   ├── constants/               # Application constants
│   ├── utils/                   # Utility functions
│   └── assets/                  # Static assets (images, fonts, videos)
│
├── public/                      # Static public assets
├── package.json                 # Frontend dependencies
├── vite.config.js               # Vite configuration
├── tailwind.config.js           # Tailwind CSS configuration
└── README.md                    # This file
```

For a detailed project structure, see [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md).

## Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

### Development Guidelines

- Follow the coding standards outlined in [CONTRIBUTING.md](./CONTRIBUTING.md)
- Use meaningful commit messages following conventional commit format
- Test your changes thoroughly before submitting a PR
- Update documentation when adding new features

### Code Organization

- **Components**: Organized by purpose (layout, ui, visuals, demo)
- **Pages**: Each page has its own folder with page-specific logic
- **Sections**: Reusable sections for homepage and other pages
- **Hooks**: Custom React hooks for shared functionality
- **Utils**: Utility functions for common operations

## Documentation

- **[PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)** - Detailed project structure and organization
- **[CONTRIBUTING.md](./CONTRIBUTING.md)** - Guidelines for contributing to the project
- **[backend/README.md](./backend/README.md)** - Backend setup and API documentation
- **[EMAILJS_SETUP.md](./EMAILJS_SETUP.md)** - EmailJS configuration guide

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](./CONTRIBUTING.md) before submitting a pull request.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add: amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- Bug fixes
- New features
- Documentation improvements
- UI/UX enhancements
- Performance optimizations
- Localization
- Testing

## License

This project is licensed under the ISC License - see the [LICENSE.md](./LICENSE.md) file for details.

## Team

This project was developed by a team of Year 3 Data Science and Computer Science students:

- **Vann Vat** - Team Leader & Data Engineer
- **Chim Panhaprasith** - Frontend Developer
- **Chhi Hangcheav** - Backend Developer
- **Toek Hengsreng** - Dataset & Research Analyst
- **Mony Meakputsoktheara** - Machine Learning Engineer
- **Phal Sovandy** - UI/UX Designer & Content Lead

## Future Roadmap

- [ ] Expand gesture vocabulary (120+ → 200+ signs)
- [ ] Improve model accuracy through better data augmentation
- [ ] Support dynamic gestures (motion-based signs)
- [ ] Add two-hand gesture recognition
- [ ] Mobile application development
- [ ] Offline functionality with TensorFlow Lite/ONNX
- [ ] Reverse translation (Text to Sign)
- [ ] Multi-language support

## Contact & Support

- **Email**: phalsovandy007@gmail.com
- **GitHub Issues**: [Report a bug or request a feature](https://github.com/Phal-Sovandy/Khmer-Sign-Language-Translation-System/issues)
- **Discord**: [Join our community](https://discord.gg/fEnEYWHz)

## Acknowledgments

- MediaPipe team for the excellent hand tracking library
- PyTorch community for the deep learning framework
- React and Vite teams for the amazing development experience
- All contributors and testers who helped improve this project

---

<div align="center">

**Made with ❤️ by the Khmer Sign Language Translation System Team**

Star this repo if you find it helpful!

</div>
