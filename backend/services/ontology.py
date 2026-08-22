"""Canonical skill ontology + weighted semantic edges.

This is the deterministic backbone of requirement↔evidence matching. Three layers:

1. ALIASES  - surface form -> canonical id. Handles "k8s"/"Kubernetes",
              "postgres"/"PostgreSQL", "GCP"/"Google Cloud Platform".
2. EDGES    - canonical -> [(related_canonical, weight)]. Handles the cases the
              JD phrases abstractly and the resume phrases concretely:
              "container orchestration" <- kubernetes (0.95).
3. CATEGORY - canonical -> bucket, used to order the SKILLS section.

Edges are directional and read as: "a JD asking for LEFT is satisfied at WEIGHT
by a candidate who has RIGHT". Weight >= 0.85 counts as a STRONG_SEMANTIC match,
0.60-0.84 PARTIAL, 0.35-0.59 WEAK.

Keeping this as data (not embeddings) means every match is explainable, testable,
and stable across runs — which matters when the output is a claim about a person.
"""

from __future__ import annotations

import re
from typing import Iterable

# --------------------------------------------------------------------------- #
# 1. Canonical skills + their alias surface forms
# --------------------------------------------------------------------------- #
# canonical -> list of alias strings (canonical itself is implied)
ALIAS_GROUPS: dict[str, list[str]] = {
    # ---- languages -------------------------------------------------------
    "java": ["java se", "java ee", "j2ee", "core java"],
    "kotlin": [],
    "scala": [],
    "python": ["python3", "py"],
    "go": ["golang", "go lang"],
    "javascript": ["js", "es6", "ecmascript", "vanilla js"],
    "typescript": ["ts"],
    "csharp": ["c#", "c sharp", ".net c#"],
    "cpp": ["c++", "cplusplus"],
    "c": [],
    "ruby": [],
    "php": [],
    "rust": [],
    "swift": [],
    "sql": ["ansi sql", "t-sql", "pl/sql", "plsql"],
    "bash": ["shell", "shell scripting", "zsh", "sh"],
    "r": [],
    "elixir": [],

    # ---- backend frameworks ---------------------------------------------
    "spring boot": ["springboot", "spring-boot"],
    "spring": ["spring framework", "spring mvc", "spring cloud"],
    "hibernate": ["jpa", "java persistence api"],
    "django": [],
    "flask": [],
    "fastapi": [],
    "express": ["express.js", "expressjs"],
    "nestjs": ["nest.js"],
    "dotnet": [".net", ".net core", "asp.net", "asp.net core"],
    "rails": ["ruby on rails"],
    "laravel": [],
    "gin": ["gin-gonic"],
    "echo framework": [],

    # ---- frontend --------------------------------------------------------
    "react": ["react.js", "reactjs"],
    "nextjs": ["next.js", "next js"],
    "vue": ["vue.js", "vuejs", "nuxt"],
    "angular": ["angularjs", "angular 2+"],
    "svelte": ["sveltekit"],
    "html": ["html5"],
    "css": ["css3", "sass", "scss", "less", "tailwind", "tailwindcss"],
    "redux": ["zustand", "mobx"],
    "webpack": ["vite", "rollup", "esbuild", "turbopack"],

    # ---- api styles ------------------------------------------------------
    "rest": ["rest api", "restful", "restful api", "rest apis", "restful services"],
    "graphql": ["apollo", "graph ql"],
    "grpc": ["g-rpc", "protocol buffers", "protobuf"],
    "websockets": ["websocket", "socket.io"],
    "openapi": ["swagger", "oas"],
    "soap": ["wsdl"],

    # ---- data stores -----------------------------------------------------
    "postgresql": ["postgres", "psql", "aurora postgres", "rds postgres"],
    "mysql": ["mariadb", "aurora mysql"],
    "oracle db": ["oracle database", "oracle 12c", "oracle 19c"],
    "sql server": ["mssql", "microsoft sql server"],
    "mongodb": ["mongo", "documentdb"],
    "cassandra": ["scylladb", "apache cassandra"],
    "dynamodb": ["dynamo db"],
    "redis": ["elasticache", "valkey"],
    "elasticsearch": ["opensearch", "elastic search", "elk"],
    "neo4j": ["graph database"],
    "clickhouse": [],
    "snowflake": [],
    "bigquery": ["big query"],
    "redshift": [],

    # ---- messaging / streaming ------------------------------------------
    "kafka": ["apache kafka", "confluent kafka", "msk"],
    "rabbitmq": ["rabbit mq", "amqp"],
    "sqs": ["amazon sqs", "simple queue service"],
    "sns": ["amazon sns"],
    "pubsub": ["pub/sub", "google pub/sub", "cloud pub/sub"],
    "kinesis": ["amazon kinesis"],
    "activemq": ["jms"],
    "nats": [],
    "flink": ["apache flink"],
    "spark": ["apache spark", "pyspark"],
    "airflow": ["apache airflow"],

    # ---- cloud -----------------------------------------------------------
    "aws": ["amazon web services", "amazon aws"],
    "gcp": ["google cloud", "google cloud platform"],
    "azure": ["microsoft azure", "azure cloud"],
    "lambda": ["aws lambda", "serverless functions"],
    "ec2": ["amazon ec2"],
    "s3": ["amazon s3", "object storage"],
    "eks": ["amazon eks"],
    "gke": ["google kubernetes engine"],
    "aks": ["azure kubernetes service"],
    "cloudformation": ["cfn"],
    "cloud run": ["google cloud run"],

    # ---- devops / platform ----------------------------------------------
    "kubernetes": ["k8s", "kube"],
    "openshift": ["red hat openshift", "ocp"],
    "docker": ["containerisation", "containerization", "containers", "podman"],
    "helm": ["helm charts"],
    "terraform": ["hcl", "opentofu"],
    "ansible": [],
    "pulumi": [],
    "jenkins": [],
    "github actions": ["gh actions"],
    "gitlab ci": ["gitlab-ci", "gitlab pipelines"],
    "circleci": ["circle ci"],
    "argocd": ["argo cd", "argo"],
    "istio": ["service mesh", "linkerd", "envoy"],
    "nginx": ["haproxy", "reverse proxy"],
    "linux": ["unix", "ubuntu", "rhel", "centos", "debian"],
    "git": ["version control", "github", "gitlab", "bitbucket"],

    # ---- observability ---------------------------------------------------
    "on call": ["on-call", "oncall", "on call rotation", "incident response"],
    "prometheus": [],
    "grafana": [],
    "datadog": ["data dog"],
    "new relic": ["newrelic"],
    "splunk": [],
    "opentelemetry": ["otel", "distributed tracing", "jaeger", "zipkin"],
    "pagerduty": [],
    "sentry": [],

    # ---- testing ---------------------------------------------------------
    "junit": ["junit5", "testng"],
    "pytest": ["unittest"],
    "jest": ["vitest", "mocha", "jasmine"],
    "cypress": ["playwright", "selenium", "e2e testing"],
    "tdd": ["test driven development", "test-driven development"],
    "contract testing": ["pact"],
    "load testing": ["jmeter", "k6", "gatling", "performance testing"],

    # ---- architecture / practices ---------------------------------------
    "microservices": ["micro-services", "microservice architecture", "micro services"],
    "distributed systems": ["distributed computing", "distributed architecture"],
    "event driven architecture": [
        "event-driven architecture", "eda", "event driven", "event-driven",
    ],
    "domain driven design": ["ddd", "domain-driven design"],
    "system design": ["software architecture", "architecture design", "solution architecture"],
    "api design": ["api development", "api integration"],
    "caching": ["cache", "caching strategies"],
    "scalability": ["high scalability", "scaling", "horizontal scaling"],
    "high availability": ["ha", "fault tolerance", "resiliency", "resilience"],
    "cicd": ["ci/cd", "ci cd", "continuous integration", "continuous delivery",
             "continuous deployment", "build pipelines"],
    "agile": ["scrum", "kanban", "sprint planning", "agile methodologies"],
    "code review": ["peer review", "pull request review"],
    "monolith to microservices": ["strangler pattern", "modernisation", "modernization"],
    "serverless": ["faas", "function as a service"],
    "multi tenancy": ["multi-tenant", "multitenant", "multi-tenancy"],

    # ---- security --------------------------------------------------------
    "oauth": ["oauth2", "oauth 2.0", "openid connect", "oidc"],
    "jwt": ["json web token"],
    "rbac": ["role based access control", "role-based access control"],
    "abac": ["attribute based access control", "lbac", "policy based authorization"],
    "authentication": ["authn", "sso", "single sign-on", "saml"],
    "authorization": ["authz", "access control", "permissions"],
    "encryption": ["tls", "ssl", "cryptography", "at-rest encryption"],
    "owasp": ["appsec", "application security", "secure coding"],
    "secrets management": ["vault", "hashicorp vault", "kms"],

    # ---- data / ai -------------------------------------------------------
    "machine learning": ["ml", "deep learning", "neural networks"],
    "llm": ["large language model", "large language models", "genai",
            "generative ai", "foundation models"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "prompt engineering": ["prompting"],
    "vector database": ["pinecone", "weaviate", "qdrant", "pgvector", "chroma"],
    "langchain": ["llamaindex", "llama index"],
    "mcp": ["model context protocol"],
    "openai api": ["gpt api"],
    "anthropic api": ["claude api"],
    "etl": ["elt", "data pipelines", "data engineering"],
    "data modeling": ["data modelling", "schema design"],
    "pandas": ["numpy", "scipy"],

    # ---- leadership / soft -----------------------------------------------
    "technical leadership": ["tech lead", "team lead", "lead engineer", "leading a team"],
    "mentoring": ["mentorship", "coaching", "onboarding engineers"],
    "stakeholder management": ["stakeholder communication", "cross-functional collaboration"],
    "hiring": ["interviewing", "recruiting engineers"],
    "project management": ["delivery management", "roadmap planning"],
    "documentation": ["technical writing", "design docs", "rfcs"],
    "communication": ["written communication", "verbal communication"],
    "ownership": ["end-to-end ownership", "autonomy"],

    # ---- domains ----------------------------------------------------------
    "fintech": ["financial services", "banking", "payments"],
    "saas": ["b2b saas", "software as a service"],
    "e-commerce": ["ecommerce", "retail tech"],
    "healthcare": ["healthtech", "hipaa"],
    "erp": ["enterprise resource planning"],
    "accounting software": ["financial planning", "fp&a", "budgeting software", "capex"],
}

# canonical -> category (drives SKILLS section grouping and ordering)
CATEGORY: dict[str, str] = {}


def _assign(cat: str, names: Iterable[str]) -> None:
    for n in names:
        CATEGORY[n] = cat


_assign("Languages", [
    "java", "kotlin", "scala", "python", "go", "javascript", "typescript",
    "csharp", "cpp", "c", "ruby", "php", "rust", "swift", "sql", "bash", "r", "elixir",
])
_assign("Frameworks", [
    "spring boot", "spring", "hibernate", "django", "flask", "fastapi", "express",
    "nestjs", "dotnet", "rails", "laravel", "gin", "echo framework",
    "react", "nextjs", "vue", "angular", "svelte", "html", "css", "redux", "webpack",
])
_assign("APIs & Protocols", ["rest", "graphql", "grpc", "websockets", "openapi", "soap"])
_assign("Databases", [
    "postgresql", "mysql", "oracle db", "sql server", "mongodb", "cassandra",
    "dynamodb", "redis", "elasticsearch", "neo4j", "clickhouse", "snowflake",
    "bigquery", "redshift",
])
_assign("Messaging & Streaming", [
    "kafka", "rabbitmq", "sqs", "sns", "pubsub", "kinesis", "activemq", "nats",
    "flink", "spark", "airflow",
])
_assign("Cloud", [
    "aws", "gcp", "azure", "lambda", "ec2", "s3", "eks", "gke", "aks",
    "cloudformation", "cloud run",
])
_assign("DevOps & Platform", [
    "kubernetes", "openshift", "docker", "helm", "terraform", "ansible", "pulumi",
    "jenkins", "github actions", "gitlab ci", "circleci", "argocd", "istio",
    "nginx", "linux", "git", "cicd", "serverless",
])
_assign("Observability", [
    "prometheus", "grafana", "datadog", "new relic", "splunk", "opentelemetry",
    "pagerduty", "sentry", "on call",
])
_assign("Testing", [
    "junit", "pytest", "jest", "cypress", "tdd", "contract testing", "load testing",
])
_assign("Architecture", [
    "microservices", "distributed systems", "event driven architecture",
    "domain driven design", "system design", "api design", "caching", "scalability",
    "high availability", "monolith to microservices", "multi tenancy",
])
_assign("Security", [
    "oauth", "jwt", "rbac", "abac", "authentication", "authorization", "encryption",
    "owasp", "secrets management",
])
_assign("AI & Data", [
    "machine learning", "llm", "rag", "prompt engineering", "vector database",
    "langchain", "mcp", "openai api", "anthropic api", "etl", "data modeling", "pandas",
])
_assign("Leadership", [
    "technical leadership", "mentoring", "stakeholder management", "hiring",
    "project management", "documentation", "communication", "ownership", "agile",
    "code review",
])
_assign("Observability", ["observability", "monitoring"])
_assign("Testing", ["unit testing"])
_assign("Databases", ["nosql"])
_assign("Domain", [
    "fintech", "saas", "e-commerce", "healthcare", "erp", "accounting software",
])

# Order in which skill categories appear when relevance ties.
CATEGORY_ORDER = [
    "Languages", "Frameworks", "Architecture", "Cloud", "DevOps & Platform",
    "Databases", "Messaging & Streaming", "APIs & Protocols", "AI & Data",
    "Security", "Observability", "Testing", "Leadership", "Domain", "Other",
]

# --------------------------------------------------------------------------- #
# 2. Semantic edges: "JD asks LEFT" is satisfied at WEIGHT by "candidate has RIGHT"
# --------------------------------------------------------------------------- #
EDGES: dict[str, list[tuple[str, float]]] = {
    "kubernetes": [("openshift", 0.92), ("eks", 0.85), ("gke", 0.85), ("aks", 0.85),
                   ("docker", 0.5), ("helm", 0.6)],
    "openshift": [("kubernetes", 0.9)],
    "docker": [("kubernetes", 0.8), ("openshift", 0.75)],
    "microservices": [("distributed systems", 0.8), ("event driven architecture", 0.7),
                      ("spring boot", 0.6), ("grpc", 0.5), ("rest", 0.5)],
    "distributed systems": [("microservices", 0.8), ("kafka", 0.65),
                            ("event driven architecture", 0.75), ("scalability", 0.6)],
    "event driven architecture": [("kafka", 0.92), ("rabbitmq", 0.85), ("sqs", 0.8),
                                  ("pubsub", 0.85), ("kinesis", 0.8), ("nats", 0.8),
                                  ("activemq", 0.75)],
    "kafka": [("event driven architecture", 0.6), ("kinesis", 0.55), ("pubsub", 0.5)],
    "rest": [("api design", 0.7), ("openapi", 0.7), ("graphql", 0.45), ("grpc", 0.45)],
    "graphql": [("rest", 0.4), ("api design", 0.6)],
    "api design": [("rest", 0.8), ("graphql", 0.7), ("grpc", 0.7), ("openapi", 0.8)],
    "cicd": [("jenkins", 0.9), ("github actions", 0.9), ("gitlab ci", 0.9),
             ("circleci", 0.9), ("argocd", 0.85)],
    "jenkins": [("cicd", 0.65)],
    "github actions": [("cicd", 0.65)],
    "terraform": [("pulumi", 0.7), ("cloudformation", 0.7), ("ansible", 0.5)],
    "cloudformation": [("terraform", 0.7)],
    "aws": [("ec2", 0.7), ("s3", 0.7), ("lambda", 0.7), ("eks", 0.75), ("dynamodb", 0.6),
            ("sqs", 0.6), ("gcp", 0.4), ("azure", 0.4)],
    "gcp": [("bigquery", 0.6), ("pubsub", 0.65), ("gke", 0.75), ("cloud run", 0.7),
            ("aws", 0.4), ("azure", 0.4)],
    "azure": [("aks", 0.75), ("aws", 0.4), ("gcp", 0.4)],
    "postgresql": [("mysql", 0.65), ("sql", 0.7), ("oracle db", 0.55), ("sql server", 0.55)],
    "mysql": [("postgresql", 0.65), ("sql", 0.7)],
    "sql": [("postgresql", 0.8), ("mysql", 0.8), ("oracle db", 0.75), ("sql server", 0.75),
            ("bigquery", 0.55)],
    "nosql": [("mongodb", 0.9), ("cassandra", 0.9), ("dynamodb", 0.9), ("redis", 0.7),
              ("elasticsearch", 0.6)],
    "caching": [("redis", 0.9), ("elasticsearch", 0.4), ("nginx", 0.35)],
    "observability": [("prometheus", 0.85), ("grafana", 0.85), ("datadog", 0.9),
                      ("opentelemetry", 0.9), ("splunk", 0.8), ("new relic", 0.85),
                      ("sentry", 0.7)],
    "monitoring": [("prometheus", 0.9), ("grafana", 0.9), ("datadog", 0.9),
                   ("new relic", 0.85), ("splunk", 0.8)],
    "spring boot": [("spring", 0.85), ("java", 0.45), ("hibernate", 0.5)],
    "spring": [("spring boot", 0.9)],
    "java": [("kotlin", 0.55), ("spring boot", 0.4)],
    "kotlin": [("java", 0.6)],
    "typescript": [("javascript", 0.8)],
    "javascript": [("typescript", 0.8)],
    "react": [("nextjs", 0.85), ("redux", 0.5)],
    "nextjs": [("react", 0.85)],
    "go": [("gin", 0.4)],
    "python": [("django", 0.4), ("flask", 0.4), ("fastapi", 0.4)],
    "tdd": [("junit", 0.6), ("pytest", 0.6), ("jest", 0.6)],
    "unit testing": [("junit", 0.9), ("pytest", 0.9), ("jest", 0.9), ("tdd", 0.85)],
    "authorization": [("rbac", 0.9), ("abac", 0.9), ("oauth", 0.6), ("authentication", 0.55)],
    "authentication": [("oauth", 0.9), ("jwt", 0.85), ("authorization", 0.55)],
    "rbac": [("abac", 0.8), ("authorization", 0.7)],
    "abac": [("rbac", 0.8), ("authorization", 0.7)],
    "llm": [("openai api", 0.85), ("anthropic api", 0.85), ("rag", 0.8),
            ("langchain", 0.8), ("prompt engineering", 0.7), ("machine learning", 0.45)],
    "rag": [("vector database", 0.8), ("llm", 0.75), ("langchain", 0.8)],
    "machine learning": [("llm", 0.45), ("pandas", 0.35)],
    "technical leadership": [("mentoring", 0.7), ("code review", 0.55),
                             ("stakeholder management", 0.5), ("project management", 0.5)],
    "mentoring": [("technical leadership", 0.6)],
    "agile": [("code review", 0.35)],
    "serverless": [("lambda", 0.9), ("cloud run", 0.8)],
    "high availability": [("scalability", 0.6), ("kubernetes", 0.5)],
    "scalability": [("distributed systems", 0.65), ("high availability", 0.6),
                    ("caching", 0.5), ("kubernetes", 0.5)],
    "etl": [("airflow", 0.85), ("spark", 0.8), ("flink", 0.75)],
    "linux": [("bash", 0.7), ("docker", 0.35)],
}

# Abstract JD phrases that are not skills themselves but map onto canonicals.
CONCEPT_ALIASES: dict[str, list[str]] = {
    "container orchestration": ["kubernetes"],
    "containerisation": ["docker"],
    "containerization": ["docker"],
    "infrastructure as code": ["terraform"],
    "iac": ["terraform"],
    "message queues": ["kafka", "rabbitmq", "sqs"],
    "message brokers": ["kafka", "rabbitmq"],
    "streaming": ["kafka", "flink", "kinesis"],
    "relational databases": ["postgresql", "mysql", "sql"],
    "rdbms": ["postgresql", "mysql", "sql"],
    "nosql databases": ["nosql"],
    "cloud platforms": ["aws", "gcp", "azure"],
    "cloud native": ["kubernetes", "docker", "microservices"],
    "backend development": ["java", "python", "go", "rest"],
    "frontend development": ["react", "typescript", "css"],
    "full stack": ["react", "typescript", "rest"],
    "web services": ["rest", "grpc"],
    "asynchronous processing": ["kafka", "rabbitmq", "event driven architecture"],
    "concurrency": ["go", "java"],
    "build tools": ["webpack", "cicd"],
    "source control": ["git"],
    "unit testing": ["unit testing"],
    "automated testing": ["junit", "pytest", "jest", "cypress"],
    "observability": ["observability"],
    "monitoring": ["monitoring"],
    "logging": ["splunk", "elasticsearch", "datadog"],
    "nosql": ["nosql"],
    "devops": ["cicd", "docker", "kubernetes", "terraform"],
    "site reliability": ["observability", "kubernetes", "high availability"],
    "sre": ["observability", "kubernetes", "high availability"],
    "data structures and algorithms": ["system design"],
    "problem solving": ["system design"],
}

# --------------------------------------------------------------------------- #
# Build the lookup index
# --------------------------------------------------------------------------- #
_SURFACE_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in ALIAS_GROUPS.items():
    _SURFACE_TO_CANONICAL[_canon] = _canon
    for _a in _aliases:
        _SURFACE_TO_CANONICAL[_a] = _canon

# Pseudo-canonicals that only exist as edge targets/sources.
for _pseudo in ("nosql", "observability", "monitoring", "unit testing"):
    _SURFACE_TO_CANONICAL.setdefault(_pseudo, _pseudo)
    CATEGORY.setdefault(_pseudo, "Other")

_PUNCT = re.compile(r"[^a-z0-9+#./\- ]+")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip decorative punctuation."""
    t = (text or "").lower().replace("’", "'").replace("&", " and ")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def canonicalise(term: str) -> str:
    """Map a surface form to its canonical id, or return the normalised term."""
    n = normalise(term)
    if n in _SURFACE_TO_CANONICAL:
        return _SURFACE_TO_CANONICAL[n]
    # try singular/plural and common suffix trimming
    for variant in (n.rstrip("s"), n + "s", n.replace("-", " "), n.replace(" ", "")):
        if variant in _SURFACE_TO_CANONICAL:
            return _SURFACE_TO_CANONICAL[variant]
    return n


def expand_concept(term: str) -> list[str]:
    """Abstract JD phrase -> concrete canonicals it can be satisfied by."""
    n = normalise(term)
    if n in CONCEPT_ALIASES:
        return CONCEPT_ALIASES[n]
    for phrase, targets in CONCEPT_ALIASES.items():
        if phrase in n:
            return targets
    return []


def related(canonical: str) -> list[tuple[str, float]]:
    return EDGES.get(canonical, [])


def category_of(canonical: str) -> str:
    return CATEGORY.get(canonical, "Other")


# Canonicals that exist only to make matching work. A JD asking for
# "observability" is satisfied by Datadog; but "Observability" is not itself a
# skill to list, and printing it next to Datadog reads as padding.
CONCEPT_ONLY = {
    "observability", "monitoring", "unit testing", "nosql", "on call",
    "authentication", "authorization", "api design", "system design",
    "scalability", "high availability", "caching", "code review",
}


def known_surface_forms() -> set[str]:
    return set(_SURFACE_TO_CANONICAL)


# --------------------------------------------------------------------------- #
# Display names
# --------------------------------------------------------------------------- #
# The canonical id is lowercase for matching; the resume needs the form a human
# writes. Only the cases plain title-casing gets wrong are listed.
DISPLAY: dict[str, str] = {
    "aws": "AWS", "gcp": "GCP", "azure": "Azure", "aks": "AKS", "eks": "EKS",
    "gke": "GKE", "ec2": "EC2", "s3": "S3", "sqs": "SQS", "sns": "SNS",
    "pubsub": "Pub/Sub", "cicd": "CI/CD", "csharp": "C#", "cpp": "C++", "c": "C",
    "r": "R", "sql": "SQL", "nosql": "NoSQL", "html": "HTML", "css": "CSS",
    "rest": "REST", "grpc": "gRPC", "graphql": "GraphQL", "soap": "SOAP",
    "openapi": "OpenAPI", "jwt": "JWT", "oauth": "OAuth 2.0", "rbac": "RBAC",
    "abac": "ABAC", "owasp": "OWASP", "etl": "ETL", "llm": "LLMs", "rag": "RAG",
    "mcp": "MCP", "tdd": "TDD", "ddd": "Domain-Driven Design", "erp": "ERP",
    "saas": "SaaS", "kafka": "Apache Kafka", "postgresql": "PostgreSQL",
    "mysql": "MySQL", "mongodb": "MongoDB", "dynamodb": "DynamoDB",
    "elasticsearch": "Elasticsearch", "clickhouse": "ClickHouse",
    "bigquery": "BigQuery", "redshift": "Redshift", "rabbitmq": "RabbitMQ",
    "activemq": "ActiveMQ", "nats": "NATS", "kinesis": "Kinesis",
    "oracle db": "Oracle Database", "sql server": "SQL Server", "neo4j": "Neo4j",
    "javascript": "JavaScript", "typescript": "TypeScript", "nodejs": "Node.js",
    "nextjs": "Next.js", "nestjs": "NestJS", "vue": "Vue.js", "react": "React",
    "dotnet": ".NET", "rails": "Ruby on Rails", "php": "PHP", "go": "Go",
    "kubernetes": "Kubernetes", "openshift": "OpenShift", "argocd": "Argo CD",
    "github actions": "GitHub Actions", "gitlab ci": "GitLab CI",
    "circleci": "CircleCI", "nginx": "NGINX", "istio": "Istio",
    "opentelemetry": "OpenTelemetry", "pagerduty": "PagerDuty", "on call": "On-Call & Incident Response",
    "new relic": "New Relic", "datadog": "Datadog", "junit": "JUnit",
    "pytest": "pytest", "jest": "Jest", "cypress": "Cypress",
    "spring boot": "Spring Boot", "spring": "Spring", "fastapi": "FastAPI",
    "express": "Express.js", "langchain": "LangChain", "pandas": "pandas",
    "openai api": "OpenAI API", "anthropic api": "Anthropic API",
    "e-commerce": "E-commerce", "fintech": "FinTech",
    "domain driven design": "Domain-Driven Design",
    "event driven architecture": "Event-Driven Architecture",
    "cloudformation": "CloudFormation", "helm": "Helm", "terraform": "Terraform",
    "ansible": "Ansible", "pulumi": "Pulumi", "jenkins": "Jenkins",
    "docker": "Docker", "redis": "Redis", "linux": "Linux", "git": "Git",
    "bash": "Bash", "spark": "Apache Spark", "flink": "Apache Flink",
    "airflow": "Apache Airflow", "snowflake": "Snowflake",
    "lambda": "AWS Lambda", "cloud run": "Cloud Run", "websockets": "WebSockets",
    "cassandra": "Apache Cassandra", "hibernate": "Hibernate", "django": "Django",
    "flask": "Flask", "svelte": "Svelte", "angular": "Angular", "redux": "Redux",
    "webpack": "Webpack", "scala": "Scala", "kotlin": "Kotlin", "rust": "Rust",
    "swift": "Swift", "ruby": "Ruby", "elixir": "Elixir", "java": "Java",
    "python": "Python", "prometheus": "Prometheus", "grafana": "Grafana",
    "splunk": "Splunk", "sentry": "Sentry", "gin": "Gin",
    "microservices": "Microservices", "serverless": "Serverless",
}

_ACRONYMS = {"api", "aws", "gcp", "sql", "ci", "cd", "ui", "ux", "ml", "ai", "sre"}


def display(canonical: str) -> str:
    """Human-facing name for a canonical skill id."""
    if canonical in DISPLAY:
        return DISPLAY[canonical]
    words = canonical.replace("-", " ").split()
    return " ".join(w.upper() if w in _ACRONYMS else w.capitalize() for w in words)


def extract_surface_forms(text: str) -> dict[str, str]:
    """canonical -> the literal string as it appears in `text`.

    Aliasing is what lets a JD asking for "container orchestration" match a resume
    saying "OpenShift". But an alias must never reach the rendered document: the
    candidate wrote "Playwright", and printing "Cypress" because both canonicalise
    to `cypress` is a false claim about a product they have never used. Callers
    that render must use this, not display().
    """
    n = " " + normalise(text) + " "
    found: dict[str, str] = {}
    for surface, canon in _SURFACE_TO_CANONICAL.items():
        pattern = rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9+#])"
        if re.search(pattern, n):
            # Prefer the longest surface form actually present — "spring boot"
            # over "spring" — and prefer the canonical's own spelling when it is
            # the one written.
            current = found.get(canon)
            if current is None:
                found[canon] = surface
            elif current == canon:
                continue          # the canonical spelling is present; keep it
            elif surface == canon or len(surface) > len(current):
                found[canon] = surface
    return found


def surface_in(text: str, canonical: str) -> str | None:
    """The literal spelling of `canonical` in `text`, or None if truly absent."""
    return extract_surface_forms(text).get(canonical)


def extract_known_terms(text: str) -> set[str]:
    """Find every canonical skill mentioned anywhere in a blob of text.

    Word-boundary matched so "r" doesn't fire on every word containing r, and
    "go" doesn't fire inside "going".
    """
    n = " " + normalise(text) + " "
    found: set[str] = set()
    for surface, canon in _SURFACE_TO_CANONICAL.items():
        if len(surface) <= 2:
            pattern = rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])"
            if re.search(pattern, n):
                found.add(canon)
        elif f" {surface} " in n or f" {surface}," in n or f" {surface}." in n:
            found.add(canon)
        elif re.search(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9+#])", n):
            found.add(canon)
    return found
