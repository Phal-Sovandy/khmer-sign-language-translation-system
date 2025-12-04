# Sign Language Translation System - Web

A web application for translating Khmer Sign Language gestures into text and speech.

## Project Structure

```
├── backend/          # Python Flask backend
│   ├── model/       # Trained ML models
│   ├── templates/   # HTML templates
│   ├── server.py    # Flask server
│   └── requirements.txt
│
├── src/             # React frontend
│   ├── components/  # Reusable components
│   │   ├── demo/   # Demo page components (VideoFeedSection, DetectedTextSection, etc.)
│   │   ├── layout/ # Layout components (Header, Footer)
│   │   ├── ui/     # UI components (Tooltip, MediaWrapper)
│   │   └── ...     # Other component categories
│   ├── pages/       # Page components
│   │   └── demo/   # Demo page (DemoPage.jsx)
│   ├── sections/    # Section components (for homepage and other pages)
│   │   └── demo/   # Homepage demo section (TryDemoSection.jsx)
│   └── ...
│
├── public/           # Static assets
└── package.json      # Frontend dependencies
```

## Frontend Setup

1. Install dependencies:

```bash
npm install
```

2. Start development server:

```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Backend Setup

See [backend/README.md](./backend/README.md) for backend setup instructions.

## Deployment

The frontend is configured for Vercel deployment. The backend can be deployed separately to a Python hosting service.
