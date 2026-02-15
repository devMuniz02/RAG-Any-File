# RAG Any File

> RAG-Any-File is the ultimate repository to turn your messy folders, complex codebases, and hidden library docs into a private, searchable AI brain. Whether it's a 100-page PDF, a chaotic venv full of source code, or a screenshot of a diagram—if it's a file, this repo can "read" it, "index" it, and "chat" with it.

> **Desktop App Available**: Run as a native desktop application using the provided executable, no browser required!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/devMuniz02/RAG-Any-File)](https://github.com/devMuniz02/RAG-Any-File/issues)
[![GitHub stars](https://img.shields.io/github/stars/devMuniz02/RAG-Any-File)](https://github.com/devMuniz02/RAG-Any-File/stargazers)

![Demo](assets/PDFRAG.gif)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Cleanup](#-cleanup)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## ✨ Features

- **File Upload & Processing**: Upload multiple files (PDF, DOCX, PPTX, XLSX, CSV, images) and automatically extract text for analysis
- **Intelligent Chat**: Ask questions about your documents and get context-aware answers using RAG (Retrieval-Augmented Generation)
- **Privacy-First Design**: Choose between Local Mode (100% private, runs on your machine) and API Mode (maximum AI intelligence)
- **File Management**: Easily manage uploaded files, view page counts, and remove documents as needed
- **Dual Interface**: Choose between web-based interface or native desktop application
- **Persistent Storage**: Uploaded files, processed data, and chat history are saved for future sessions
- **Source Citations**: Answers include clickable links to relevant document sections

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- For Local Mode: A local LLM server (e.g., LM Studio, Ollama, or similar) running on `http://localhost:1234/v1/chat/completions`
- For API Mode: Internet connection and API access to LLM providers

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/devMuniz02/RAG-Any-File.git

# Navigate to the project directory
cd RAG-Any-File

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Getting Started

After cloning this repository, follow these steps to set up and run the application:

## 📁 Project Structure

```
RAG-Any-File/
├── assets/                 # Static assets (images, icons, etc.)
├── data/                   # Data files and datasets
├── desktop_app.py          # Desktop application launcher
├── docs/                   # Documentation files
├── notebooks/              # Jupyter notebooks for analysis and prototyping
├── scripts/                # Utility scripts and automation tools
├── src/                    # Source code
├── tests/                  # Unit tests and test files
├── LICENSE                 # License file
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

### Directory Descriptions

- **`assets/`**: Static assets (images, icons, etc.)
- **`data/`**: Processed document data, FAISS index, and metadata
- **`desktop_app.py`**: Script to launch the native desktop application
- **`docs/`**: Documentation files
- **`notebooks/`**: Jupyter notebooks for analysis and prototyping
- **`scripts/`**: Utility scripts and automation tools
- **`src/`**: Main source code (Flask app and RAG processor)
- **`tests/`**: Unit tests and test files
- **`uploads/`**: Uploaded files (created automatically)
- **`LICENSE`**: MIT License file
- **`README.md`**: Project documentation
- **`requirements.txt`**: Python dependencies
- **`cleanup.py`**: Script to clear all processed data and uploaded files

## 📖 Usage

### Option 1: Desktop Application (Recommended)

For the best user experience, use the native desktop app:

```bash
# Build the desktop executable (one-time setup)
pip install pyinstaller
pyinstaller --onefile --noconsole --add-data "src;src" --add-data "data;data" --add-data "uploads;uploads" desktop_app.py

# Run the desktop app
./dist/desktop_app.exe  # On Windows
```

The desktop app provides a native window interface with persistent chat history and seamless file access.

### Option 2: Web Application

For development or web access:

```bash
# Navigate to the src directory
cd src

# Run the Flask application
python app.py
```

The application will start on `http://localhost:8000`

### Basic Usage

1. **Upload Files**: Click the "Upload Files" button and select one or more files
2. **Select Files**: Choose which uploaded files to include in your chat session
3. **Configure Mode**: 
   - **Local Mode**: Uses your local LLM server for maximum privacy
   - **API Mode**: Connects to external LLM APIs for enhanced capabilities
4. **Start Chatting**: Ask questions about your documents in natural language

### Advanced Usage

- **File Management**: View uploaded files, see page counts, and remove files you no longer need
- **Persistent Sessions**: Your uploaded files and processed data are saved between sessions
- **Source Citations**: Click on citation links in responses to view relevant document sections
- **Cleanup**: Use `python cleanup.py` to clear all processed data and uploaded files

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory for custom configuration:

```env
# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True

# LLM API Configuration (for API Mode)
API_URL=https://api.openai.com/v1/chat/completions
MODEL=gpt-3.5-turbo
API_KEY=your_api_key_here

# Local LLM Configuration (for Local Mode)
LOCAL_API_URL=http://localhost:1234/v1/chat/completions
LOCAL_MODEL=local-model
```

### Application Settings

- **Upload Limit**: Maximum 100MB per file
- **Supported Formats**: PDF, DOCX, PPTX, XLSX, CSV, and image files (JPG, PNG, etc.)
- **Chunk Size**: 1000 characters with 200 character overlap for text processing
- **Embedding Model**: Uses `all-MiniLM-L6-v2` for document embeddings

## � Cleanup

To clear all processed data and uploaded files:

```bash
python cleanup.py
```

This will remove:
- FAISS index and document embeddings
- Processed document metadata
- All uploaded files

**Warning**: This action cannot be undone. Use with caution.

## �🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies (if available)
pip install -r requirements.txt

# Run the application in development mode
cd src
python app.py

# For testing file processing
# Place test files in the uploads/ directory
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **GitHub:** [https://github.com/devMuniz02/](https://github.com/devMuniz02/)
- **LinkedIn:** [https://www.linkedin.com/in/devmuniz](https://www.linkedin.com/in/devmuniz)
- **Hugging Face:** [https://huggingface.co/manu02](https://huggingface.co/manu02)
- **Portfolio:** [https://devmuniz02.github.io/](https://devmuniz02.github.io/)

Project Link: [https://github.com/devMuniz02/RAG-Any-File](https://github.com/devMuniz02/RAG-Any-File)

---

⭐ If you find this project helpful, please give it a star!
