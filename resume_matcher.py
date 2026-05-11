
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import math


class SkillNormalizer:
    
    def __init__(self):
        self.skill_mappings = {
            
            'python': ['python', 'py', 'python3', 'python 3'],
            'java': ['java', 'j2ee'],
            'javascript': ['javascript', 'js', 'node', 'node.js', 'nodejs'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'kotlin': ['kotlin', 'kotlin.js'],
            'r': ['r programming', 'r language', 'r'],
            
            'react': ['react', 'react.js', 'reactjs'],
            'vue': ['vue', 'vue.js', 'vuejs'],
            'html': ['html', 'html5'],
            'css': ['css', 'css3'],
            'rest': ['rest', 'rest apis', 'restful', 'restful apis', 'api'],
            
            # Frameworks & Libraries
            'tensorflow': ['tensorflow', 'tf'],
            'pytorch': ['pytorch', 'torch'],
            'spring boot': ['spring boot', 'spring', 'springboot'],
            'django': ['django'],
            'flask': ['flask'],
            'pandas': ['pandas'],
            'numpy': ['numpy'],
            'scikit-learn': ['scikit-learn', 'sklearn'],
            
            # Databases
            'mysql': ['mysql'],
            'postgresql': ['postgresql', 'postgres', 'postgre'],
            'mongodb': ['mongodb', 'mongo'],
            'firebase': ['firebase'],
            'sqlite': ['sqlite'],
            'sql': ['sql'],
            
            # Cloud & DevOps
            'aws': ['aws', 'amazon web services', 'amazon aws'],
            'azure': ['azure', 'microsoft azure'],
            'docker': ['docker', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s'],
            'ci/cd': ['ci/cd', 'cicd', 'continuous integration', 'continuous deployment'],
            
            # Data & ML
            'machine learning': ['machine learning', 'ml', 'machine-learning'],
            'deep learning': ['deep learning', 'deep-learning'],
            'data science': ['data science', 'data-science'],
            'nlp': ['nlp', 'natural language processing'],
            'computer vision': ['computer vision', 'cv'],
            'bert': ['bert'],
            'transformers': ['transformers'],
            
            # Testing & QA
            'selenium': ['selenium'],
            'automation testing': ['automation testing', 'test automation'],
            'jira': ['jira'],
            
            # Other Skills
            'git': ['git', 'github', 'gitlab'],
            'agile': ['agile', 'scrum', 'sprint'],
            'gpu': ['gpu', 'graphics processing', 'cuda'],
        }
        
        # Build reverse mapping for normalization
        self.reverse_map = {}
        for normalized, variants in self.skill_mappings.items():
            for variant in variants:
                self.reverse_map[variant.lower()] = normalized
    
    def normalize(self, text: str) -> List[str]:
        # Convert to lowercase and split
        words = re.findall(r'\b[\w\+\-/]+\b', text.lower())
        
        normalized_skills = set()
        i = 0
        while i < len(words):
            # Check two-word combinations first
            if i < len(words) - 1:
                two_word = f"{words[i]} {words[i+1]}"
                if two_word in self.reverse_map:
                    normalized_skills.add(self.reverse_map[two_word])
                    i += 2
                    continue
            
            # Check single word
            word = words[i]
            if word in self.reverse_map:
                normalized_skills.add(self.reverse_map[word])
            
            i += 1
        
        return sorted(list(normalized_skills))


class TFIDFVectorizer:
    
    def __init__(self, documents: List[List[str]]):

        self.documents = documents
        self.vocabulary = self._build_vocabulary()
        self.idf = self._compute_idf()
    
    def _build_vocabulary(self) -> Dict[str, int]:
        vocab = {}
        idx = 0
        for doc in self.documents:
            for skill in doc:
                if skill not in vocab:
                    vocab[skill] = idx
                    idx += 1
        return vocab
    
    def _compute_idf(self) -> Dict[str, float]:
        idf = {}
        total_docs = len(self.documents)
        
        for skill in self.vocabulary:
            doc_count = sum(1 for doc in self.documents if skill in doc)
            idf[skill] = math.log(total_docs / (1 + doc_count))
        
        return idf
    
    def vectorize(self, document: List[str]) -> Dict[str, float]:
        """Convert skill document to TF-IDF vector."""
        vector = {}
        
        # Compute term frequency
        skill_count = Counter(document)
        total_skills = len(document)
        
        for skill, count in skill_count.items():
            if skill in self.vocabulary:
                # TF = (count / total)
                tf = count / total_skills if total_skills > 0 else 0
                # TF-IDF = TF * IDF
                tfidf = tf * self.idf.get(skill, 0)
                vector[skill] = tfidf
        
        return vector


def compute_cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two vectors."""
    # Get all unique keys
    all_keys = set(vec1.keys()) | set(vec2.keys())
    
    if not all_keys:
        return 0.0
    
    # Compute dot product
    dot_product = sum(vec1.get(key, 0) * vec2.get(key, 0) for key in all_keys)
    
    # Compute magnitudes
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


def load_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load resumes and job descriptions from JSON files."""
    with open(f"{data_dir}/resumes.json", 'r') as f:
        resumes = json.load(f)
    
    with open(f"{data_dir}/job_descriptions.json", 'r') as f:
        job_descriptions = json.load(f)
    
    return resumes, job_descriptions


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("RESUME-TO-JOB MATCHING SYSTEM")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    data_dir = Path(__file__).parent / "data"
    resumes, job_descriptions = load_data(str(data_dir))
    print(f"✓ Loaded {len(resumes)} resumes and {len(job_descriptions)} job descriptions")
    
    # Step 2: Normalize resume skills
    print("\n[2/5] Normalizing resume skills...")
    normalizer = SkillNormalizer()
    resume_skills = []
    
    for resume in resumes:
        normalized = normalizer.normalize(resume['skills'])
        resume_skills.append(normalized)
        print(f"  {resume['name']}: {len(normalized)} normalized skills")
    
    # Step 3: Compute TF-IDF vectors for resumes
    print("\n[3/5] Computing TF-IDF vectors for resumes...")
    vectorizer = TFIDFVectorizer(resume_skills)
    resume_vectors = [vectorizer.vectorize(skills) for skills in resume_skills]
    print(f"✓ Vocabulary size: {len(vectorizer.vocabulary)} unique skills")
    
    # Step 4: Create binary vectors for job descriptions
    print("\n[4/5] Creating binary vectors for job descriptions...")
    jd_skills = []
    jd_vectors = []
    
    for jd in job_descriptions:
        normalized = normalizer.normalize(jd['skills'])
        jd_skills.append(normalized)
        
        # Create binary vector (1 if skill present, 0 otherwise)
        binary_vector = {skill: 1.0 for skill in normalized}
        jd_vectors.append(binary_vector)
        print(f"  {jd['company']} - {jd['position']}: {len(normalized)} required skills")
    
    # Step 5: Calculate cosine similarity and find top matches
    print("\n[5/5] Calculating matches and generating results...\n")
    
    results = []
    
    for jd_idx, (jd, jd_vector) in enumerate(zip(job_descriptions, jd_vectors)):
        print("=" * 70)
        print(f"JOB DESCRIPTION #{jd_idx + 1}")
        print(f"Company: {jd['company']}")
        print(f"Position: {jd['position']}")
        print(f"Required Skills: {', '.join(jd_skills[jd_idx])}")
        print("=" * 70)
        
        # Calculate similarity for all resumes
        similarities = []
        for resume_idx, (resume, resume_vector) in enumerate(zip(resumes, resume_vectors)):
            similarity = compute_cosine_similarity(resume_vector, jd_vector)
            similarities.append({
                'resume_id': resume['id'],
                'name': resume['name'],
                'university': resume['university'],
                'similarity': similarity,
                'matched_skills': [s for s in jd_skills[jd_idx] if s in resume_skills[resume_idx]]
            })
        
        # Sort by similarity and get top 3
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_3 = similarities[:3]
        
        # Display results
        for rank, match in enumerate(top_3, 1):
            print(f"\n🏆 RANK #{rank}: {match['name']} ({match['resume_id']})")
            print(f"   University: {match['university']}")
            print(f"   Match Score: {match['similarity']:.4f}")
            print(f"   Matched Skills ({len(match['matched_skills'])}): {', '.join(match['matched_skills'][:5])}")
            if len(match['matched_skills']) > 5:
                print(f"                and {len(match['matched_skills']) - 5} more...")
        
        results.append({
            'job_id': jd['id'],
            'company': jd['company'],
            'position': jd['position'],
            'top_3_matches': top_3
        })
        print("\n")
    
    # Step 6: Export results to JSON
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "matching_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 70)
    print(f"✓ Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
