#!/usr/bin/env python3
"""Test Cross-File Intelligence System - relationship mapping and danger zones"""

import sys
import os
import json

# Direct import to avoid dependency issues
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import the modules directly
import importlib.util

# Load Deep Understanding Agent
spec1 = importlib.util.spec_from_file_location(
    "deep_understanding_agent",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                 "src/codebaseiq/agents/deep_understanding_agent.py")
)
deep_module = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(deep_module)
DeepUnderstandingAgent = deep_module.DeepUnderstandingAgent
CodeContext = deep_module.CodeContext

# Load Cross-File Intelligence
spec2 = importlib.util.spec_from_file_location(
    "cross_file_intelligence",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                 "src/codebaseiq/agents/cross_file_intelligence.py")
)
cross_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(cross_module)
CrossFileIntelligence = cross_module.CrossFileIntelligence

def test_cross_file_intelligence():
    """Test relationship mapping and danger zone identification"""
    
    print("🔗 Testing Cross-File Intelligence System...")
    
    # Step 1: Create a simulated codebase with Deep Understanding
    deep_agent = DeepUnderstandingAgent()
    
    # Simulate file contents
    file_contents = {
        "config/settings.py": '''
"""Application configuration - CRITICAL FILE
Changes here affect the entire application
"""
import os
from dotenv import load_dotenv

class Config:
    """Central configuration for all services"""
    
    # Database settings
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://localhost/app')
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    
    # API settings
    API_KEY = os.environ.get('API_KEY')
    API_RATE_LIMIT = 1000
    
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    JWT_EXPIRY = 3600
    
    # Feature flags
    ENABLE_CACHE = True
    ENABLE_ANALYTICS = True

config = Config()
''',
        
        "auth/authentication.py": '''
"""Authentication service - handles user login and JWT tokens"""
from config.settings import config
from database.models import User
from utils.crypto import hash_password, verify_password
import jwt

class AuthService:
    """Core authentication service used by all API endpoints"""
    
    def __init__(self):
        self.secret = config.SECRET_KEY
        self.jwt_expiry = config.JWT_EXPIRY
        
    def login(self, email, password):
        """Authenticate user and return JWT token"""
        user = User.query.filter_by(email=email).first()
        if user and verify_password(password, user.password_hash):
            token = self.generate_token(user)
            return {"token": token, "user_id": user.id}
        return None
        
    def generate_token(self, user):
        """Generate JWT token for authenticated user"""
        payload = {
            "user_id": user.id,
            "email": user.email,
            "exp": datetime.utcnow() + timedelta(seconds=self.jwt_expiry)
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")
        
    def verify_token(self, token):
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
''',

        "api/users.py": '''
"""User API endpoints"""
from flask import Blueprint, request, jsonify
from auth.authentication import AuthService
from database.models import User, db
from decorators.auth import require_auth

users_bp = Blueprint('users', __name__)
auth_service = AuthService()

@users_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.get_json()
    result = auth_service.login(data['email'], data['password'])
    if result:
        return jsonify(result), 200
    return jsonify({"error": "Invalid credentials"}), 401

@users_bp.route('/profile', methods=['GET'])
@require_auth
def get_profile(current_user):
    """Get user profile - requires authentication"""
    return jsonify({
        "id": current_user.id,
        "email": current_user.email,
        "created_at": current_user.created_at
    })

@users_bp.route('/users', methods=['GET'])
@require_auth
def list_users(current_user):
    """List all users - admin only"""
    if not current_user.is_admin:
        return jsonify({"error": "Admin access required"}), 403
    
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])
''',

        "api/products.py": '''
"""Product API endpoints"""
from flask import Blueprint, request, jsonify
from database.models import Product, db
from decorators.auth import require_auth
from cache.redis_cache import cache

products_bp = Blueprint('products', __name__)

@products_bp.route('/products', methods=['GET'])
@cache.cached(timeout=300)
def list_products():
    """List all products - public endpoint with caching"""
    products = Product.query.all()
    return jsonify([p.to_dict() for p in products])

@products_bp.route('/products', methods=['POST'])
@require_auth
def create_product(current_user):
    """Create new product - requires authentication"""
    if not current_user.is_admin:
        return jsonify({"error": "Admin access required"}), 403
        
    data = request.get_json()
    product = Product(**data)
    db.session.add(product)
    db.session.commit()
    
    # Clear cache
    cache.delete('products_list')
    
    return jsonify(product.to_dict()), 201
''',

        "database/models.py": '''
"""Database models - core data structures"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from database import db

class User(db.Model):
    """User model - used throughout the application"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat()
        }

class Product(db.Model):
    """Product model"""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    price = Column(Integer, nullable=False)
    stock = Column(Integer, default=0)
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "stock": self.stock
        }
''',

        "decorators/auth.py": '''
"""Authentication decorators used by API endpoints"""
from functools import wraps
from flask import request, jsonify
from auth.authentication import AuthService

auth_service = AuthService()

def require_auth(f):
    """Decorator to require authentication for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({"error": "No token provided"}), 401
            
        payload = auth_service.verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
            
        # Get user from database
        from database.models import User
        current_user = User.query.get(payload['user_id'])
        if not current_user:
            return jsonify({"error": "User not found"}), 404
            
        return f(current_user, *args, **kwargs)
    
    return decorated_function
''',

        "utils/crypto.py": '''
"""Cryptographic utilities"""
import bcrypt

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hash):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hash)
''',

        "cache/redis_cache.py": '''
"""Redis cache configuration"""
from flask_caching import Cache
from config.settings import config

cache = Cache(config={
    'CACHE_TYPE': 'redis' if config.ENABLE_CACHE else 'simple',
    'CACHE_REDIS_URL': config.REDIS_URL
})
''',

        "main.py": '''
"""Main application entry point"""
from flask import Flask
from config.settings import config
from api.users import users_bp
from api.products import products_bp
from database import db
from cache.redis_cache import cache

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URL
    
    # Initialize extensions
    db.init_app(app)
    cache.init_app(app)
    
    # Register blueprints
    app.register_blueprint(users_bp, url_prefix='/api')
    app.register_blueprint(products_bp, url_prefix='/api')
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
'''
    }
    
    # Analyze each file with Deep Understanding
    print("\n📊 Phase 1: Deep Understanding Analysis")
    contexts = {}
    for file_path, content in file_contents.items():
        context = deep_agent.analyze_file(file_path, content)
        contexts[file_path] = context
        print(f"  ✓ Analyzed: {file_path} - Risk: {context.modification_risk}")
    
    # Step 2: Run Cross-File Intelligence Analysis
    print("\n🔗 Phase 2: Cross-File Intelligence Analysis")
    cross_intel = CrossFileIntelligence()
    results = cross_intel.analyze_relationships(contexts, file_contents)
    
    # Display results
    print("\n📈 Import Graph:")
    for file, imports in results['import_graph'].items():
        if imports:
            print(f"  {file} imports: {', '.join(list(imports)[:3])}")
    
    print("\n🔄 Reverse Dependencies (who depends on each file):")
    for file, dependents in results['reverse_dependencies'].items():
        if dependents:
            print(f"  {file} ← used by: {', '.join(dependents)}")
    
    print("\n⚠️  Impact Zones (danger zones):")
    high_impact_files = []
    for file, zone in results['impact_zones'].items():
        if zone['risk_level'] in ['CRITICAL', 'HIGH']:
            high_impact_files.append(file)
            print(f"  {file}:")
            print(f"    Risk Level: {zone['risk_level']}")
            print(f"    Direct Impact: {zone['direct_impact']}")
            print(f"    Total Impact: {zone['total_impact']} files")
            print(f"    AI Warning: {zone['ai_warning']}")
            print(f"    Strategy: {zone['modification_strategy']}")
    
    print("\n🚨 Critical Interfaces:")
    for interface in results['critical_interfaces'][:5]:
        print(f"  {interface['file']}:")
        print(f"    Type: {interface['interface_type']}")
        print(f"    Dependents: {interface['dependent_count']} files")
        print(f"    Stability Score: {interface['stability_score']}/100")
        print(f"    Guidance: {interface['ai_guidance']}")
    
    print("\n🔄 Circular Dependencies:")
    if results['circular_dependencies']:
        for cycle in results['circular_dependencies']:
            print(f"  Cycle: {' → '.join(cycle)}")
    else:
        print("  None found (good!)")
    
    print("\n🤖 AI Modification Guidance Summary:")
    ai_guidance = results['ai_modification_guidance']
    print(f"  Total Files: {ai_guidance['summary']['total_files']}")
    print(f"  High Risk Files: {ai_guidance['summary']['high_risk_files']}")
    print(f"  Critical Interfaces: {ai_guidance['summary']['critical_interfaces']}")
    
    print("\n📋 Risk Categories:")
    print(f"  DO NOT MODIFY: {len(ai_guidance['risk_categories']['do_not_modify'])} files")
    for item in ai_guidance['risk_categories']['do_not_modify'][:3]:
        print(f"    - {item['file']}: {item['reason']}")
    
    print(f"\n  EXTREME CAUTION: {len(ai_guidance['risk_categories']['extreme_caution'])} files")
    for item in ai_guidance['risk_categories']['extreme_caution'][:3]:
        print(f"    - {item['file']}: {item['impact_count']} dependencies")
    
    print("\n📝 Quick Reference for AI:")
    print(ai_guidance['quick_reference'])
    
    # Test specific scenarios
    print("\n🧪 Testing Specific Scenarios:")
    
    # Scenario 1: What happens if we modify config/settings.py?
    print("\n1. Impact of modifying config/settings.py:")
    config_impact = results['impact_zones'].get('config/settings.py', {})
    print(f"   Risk: {config_impact.get('risk_level', 'N/A')}")
    print(f"   Files affected: {config_impact.get('total_impact', 0)}")
    print(f"   Warning: {config_impact.get('ai_warning', 'N/A')}")
    
    # Scenario 2: What about auth/authentication.py?
    print("\n2. Impact of modifying auth/authentication.py:")
    auth_impact = results['impact_zones'].get('auth/authentication.py', {})
    print(f"   Risk: {auth_impact.get('risk_level', 'N/A')}")
    print(f"   Files affected: {auth_impact.get('total_impact', 0)}")
    print(f"   Warning: {auth_impact.get('ai_warning', 'N/A')}")
    
    # Validate results
    assert len(results['import_graph']) > 0, "Import graph should not be empty"
    assert len(results['reverse_dependencies']) > 0, "Should have reverse dependencies"
    assert len(results['impact_zones']) > 0, "Should have impact zones"
    assert len(high_impact_files) > 0, "Should identify high-risk files"
    assert 'config/settings.py' in high_impact_files, "Config should be high risk"
    assert len(results['critical_interfaces']) > 0, "Should identify critical interfaces"
    
    print("\n✅ Cross-File Intelligence System Test Passed!")
    print("✅ Successfully identifies danger zones and provides AI safety guidance")
    
    # Save results for inspection
    output = {
        "high_impact_files": high_impact_files,
        "ai_guidance": ai_guidance,
        "sample_impact_zone": config_impact,
        "critical_interfaces": results['critical_interfaces'][:3]
    }
    
    with open("test_cross_file_output.json", "w") as f:
        json.dump(output, f, indent=2)
        print("\n📄 Detailed results saved to test_cross_file_output.json")

if __name__ == "__main__":
    test_cross_file_intelligence()