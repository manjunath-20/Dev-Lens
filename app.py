from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────

JOB_DESCRIPTION_PROMPT = """
You are a senior tech career advisor and hiring expert with 15+ years of experience.
Analyze the given job description and return ONLY valid JSON with NO extra text, markdown, or code fences.

Return ONLY valid JSON in this exact structure:
{
  "role_title": "string – inferred job title",
  "seniority": "Junior | Mid | Senior | Lead | Principal",
  "languages": [{"name": "string", "priority": "Must | Good to have", "reason": "string"}],
  "technologies": [{"name": "string", "category": "Framework | Cloud | Database | DevOps | Tool | Library", "priority": "Must | Good to have"}],
  "concepts": ["string list of key concepts/paradigms to learn"],
  "resume_tips": ["string list of specific resume advice tailored to this JD"],
  "skills_gap_roadmap": ["string list – ordered learning roadmap steps"],
  "interview_topics": ["string list of likely interview topics"],
  "salary_range": "string – estimated range based on role/seniority",
  "summary": "string – 2-sentence overview of what this role demands"
}
"""

PROJECT_ANALYZER_PROMPT = """
You are a principal software engineer reviewing a project structure like you're doing a thorough technical audit.
Analyze the given project structure/module list and return ONLY valid JSON with NO extra text, markdown, or code fences.

Return ONLY valid JSON in this exact structure:
{
  "project_name": "string – inferred project name",
  "project_type": "string – e.g. REST API, Microservice, Monolith, CLI Tool, Data Pipeline, etc.",
  "domain": "string – business domain e.g. E-commerce, FinTech, DevOps Platform, etc.",
  "languages": [{"name": "string", "confidence": "High | Medium | Low", "evidence": "string"}],
  "frameworks": [{"name": "string", "role": "string"}],
  "architecture_pattern": "string – e.g. MVC, Clean Architecture, Hexagonal, Event-Driven, etc.",
  "databases": ["string list of likely databases used"],
  "key_modules": [{"module": "string", "purpose": "string", "complexity": "Low | Medium | High"}],
  "code_quality_signals": ["string list of positive signals or red flags observed"],
  "technical_debt_risks": ["string list of potential risk areas"],
  "missing_essentials": ["string list of important missing pieces like tests, CI/CD, docs"],
  "senior_recommendations": ["string list of actionable improvements a senior dev would recommend"],
  "estimated_team_size": "string – e.g. Solo, 2-5 devs, 5-10 devs",
  "maturity": "Prototype | Early Stage | Production | Legacy",
  "summary": "string – 2-3 sentence senior dev overview of the project"
}
"""


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze-job", methods=["POST"])
def analyze_job():
    data = request.get_json()
    job_description = data.get("job_description", "").strip()

    if not job_description:
        return jsonify({"error": "Job description is required."}), 400
    if len(job_description) < 50:
        return jsonify({"error": "Job description is too short. Please provide more detail."}), 400

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": JOB_DESCRIPTION_PROMPT},
                {"role": "user",   "content": f"Analyze this job description:\n\n{job_description}"}
            ],
            temperature=0.4,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return jsonify({"success": True, "data": result})

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/analyze-project", methods=["POST"])
def analyze_project():
    project_text = ""

    if request.content_type and "multipart/form-data" in request.content_type:
        file = request.files.get("file")
        if file:
            project_text = file.read().decode("utf-8", errors="ignore")
        extra = request.form.get("extra_context", "")
        if extra:
            project_text = extra + "\n\n" + project_text
    else:
        data = request.get_json()
        project_text = data.get("project_structure", "").strip()

    if not project_text:
        return jsonify({"error": "Project structure is required."}), 400
    if len(project_text) < 30:
        return jsonify({"error": "Project structure is too short. Please provide more detail."}), 400

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": PROJECT_ANALYZER_PROMPT},
                {"role": "user",   "content": f"Analyze this project structure:\n\n{project_text}"}
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return jsonify({"success": True, "data": result})

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)