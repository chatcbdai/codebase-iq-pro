#!/usr/bin/env python3
"""
Comprehensive Integration Test for Enhanced CodebaseIQ Pro
Tests the full pipeline from analysis to AI knowledge packaging
"""

import asyncio
import json
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import server components
from codebaseiq.server import CodebaseIQProServer
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions

# Simulate server environment
os.environ['OPENAI_API_KEY'] = 'test-key-12345'

class TestCodebaseIQIntegration:
    """Test the full enhanced CodebaseIQ pipeline"""
    
    def __init__(self):
        self.test_dir = None
        self.server = None
        
    def create_test_codebase(self) -> Path:
        """Create a test codebase with various patterns"""
        self.test_dir = tempfile.mkdtemp(prefix="codebaseiq_test_")
        print(f"📁 Created test directory: {self.test_dir}")
        
        # Create directory structure
        paths = [
            "src/models",
            "src/services",
            "src/api",
            "src/utils",
            "tests"
        ]
        
        for path in paths:
            os.makedirs(os.path.join(self.test_dir, path), exist_ok=True)
            
        # Create test files
        test_files = {
            "src/models/user.py": '''
"""User model - CRITICAL for authentication"""
from sqlalchemy import Column, Integer, String, Boolean
from database import Base

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    
    def verify_password(self, password):
        """SECURITY: Verify user password"""
        return check_password_hash(self.password_hash, password)
        
    def has_permission(self, action):
        """CRITICAL: Permission checking"""
        if not self.is_active:
            return False
        return action in self.permissions
''',
            
            "src/services/auth_service.py": '''
"""Authentication service - handles login/logout"""
from models.user import User
from utils.security import generate_token, verify_token
import logging

logger = logging.getLogger(__name__)

class AuthService:
    """CRITICAL: Authentication service - DO NOT MODIFY WITHOUT REVIEW"""
    
    def __init__(self):
        self.max_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        
    def authenticate(self, email, password):
        """
        Authenticate user and return JWT token
        SECURITY: Rate limited to prevent brute force
        """
        # Check rate limiting
        if self.is_locked_out(email):
            logger.warning(f"Login attempt for locked account: {email}")
            raise SecurityError("Account temporarily locked")
            
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.verify_password(password):
            self.record_failed_attempt(email)
            return None
            
        # Clear failed attempts on success
        self.clear_failed_attempts(email)
        
        # Generate JWT token
        token = generate_token(user.id)
        logger.info(f"User {email} authenticated successfully")
        
        return {
            "token": token,
            "user_id": user.id,
            "email": user.email
        }
        
    def logout(self, token):
        """Invalidate user token"""
        # Add token to blacklist
        blacklist_token(token)
        return True
''',
            
            "src/services/payment_service.py": '''
"""Payment processing service"""
import stripe
from models.order import Order
from utils.encryption import encrypt_sensitive_data
import logging

logger = logging.getLogger(__name__)

class PaymentService:
    """Handles payment processing - PCI compliance required"""
    
    def __init__(self):
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
        
    def process_payment(self, order: Order, payment_method: dict):
        """
        Process payment for an order
        COMPLIANCE: PCI DSS - Never log credit card details
        """
        try:
            # CRITICAL: Never store raw credit card data
            encrypted_method = encrypt_sensitive_data(payment_method)
            
            # Create charge
            charge = stripe.Charge.create(
                amount=int(order.total * 100),  # Convert to cents
                currency='usd',
                source=payment_method['token'],
                description=f"Order {order.id}"
            )
            
            # Log success without sensitive data
            logger.info(f"Payment processed for order {order.id}")
            
            return {
                "success": True,
                "transaction_id": charge.id,
                "amount": order.total
            }
            
        except stripe.error.CardError as e:
            logger.error(f"Payment failed for order {order.id}: {e.user_message}")
            return {
                "success": False,
                "error": e.user_message
            }
''',
            
            "src/api/user_api.py": '''
"""User API endpoints"""
from flask import Blueprint, request, jsonify
from services.auth_service import AuthService
from utils.decorators import require_auth, validate_input
from models.user import User

user_bp = Blueprint('users', __name__)
auth_service = AuthService()

@user_bp.route('/login', methods=['POST'])
@validate_input(['email', 'password'])
def login():
    """User login endpoint"""
    data = request.get_json()
    
    result = auth_service.authenticate(
        email=data['email'],
        password=data['password']
    )
    
    if result:
        return jsonify(result), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401
        
@user_bp.route('/profile', methods=['GET'])
@require_auth
def get_profile(current_user):
    """Get user profile - requires authentication"""
    return jsonify({
        "id": current_user.id,
        "email": current_user.email,
        "created_at": current_user.created_at
    })
    
@user_bp.route('/update', methods=['PUT'])
@require_auth
def update_profile(current_user):
    """Update user profile"""
    data = request.get_json()
    
    # Only allow updating certain fields
    allowed_fields = ['first_name', 'last_name', 'phone']
    
    for field in allowed_fields:
        if field in data:
            setattr(current_user, field, data[field])
            
    db.session.commit()
    
    return jsonify({"message": "Profile updated successfully"})
''',
            
            "src/utils/security.py": '''
"""Security utilities - CRITICAL"""
import jwt
import hashlib
from datetime import datetime, timedelta

# CRITICAL: JWT secret - never expose
JWT_SECRET = os.getenv('JWT_SECRET', 'development-only-secret')

def generate_token(user_id: int) -> str:
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    
def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")
        
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
''',
            
            "tests/test_auth.py": '''
"""Tests for authentication service"""
import pytest
from services.auth_service import AuthService
from models.user import User

class TestAuthService:
    def test_successful_login(self):
        """Test successful authentication"""
        service = AuthService()
        result = service.authenticate("test@example.com", "password123")
        assert result is not None
        assert 'token' in result
        
    def test_failed_login(self):
        """Test failed authentication"""
        service = AuthService()
        result = service.authenticate("test@example.com", "wrongpassword")
        assert result is None
'''
        }
        
        # Write all files
        for file_path, content in test_files.items():
            full_path = os.path.join(self.test_dir, file_path)
            with open(full_path, 'w') as f:
                f.write(content)
                
        print(f"✅ Created {len(test_files)} test files")
        return Path(self.test_dir)
        
    async def test_analyze_codebase(self) -> Dict[str, Any]:
        """Test the analyze_codebase tool"""
        print("\n🔍 Testing analyze_codebase...")
        
        result = await self.server._analyze_codebase(
            path=str(self.test_dir),
            analysis_type="full",
            enable_embeddings=False  # Skip embeddings for test
        )
        
        assert result['status'] == 'success', "Analysis should succeed"
        assert result['files_analyzed'] > 0, "Should analyze files"
        assert 'enhanced_understanding' in result['results'], "Should have enhanced understanding"
        assert result['ai_ready'] == True, "Should be AI ready"
        
        print(f"  ✓ Analyzed {result['files_analyzed']} files")
        print(f"  ✓ AI Ready: {result['ai_ready']}")
        print(f"  ✓ Instant Context Preview: {result['instant_context'][:100]}...")
        
        return result
        
    async def test_get_ai_knowledge_package(self):
        """Test retrieving AI knowledge package"""
        print("\n📦 Testing get_ai_knowledge_package...")
        
        result = await self.server._get_ai_knowledge_package()
        
        assert 'error' not in result, "Should not have errors"
        assert 'instant_context' in result, "Should have instant context"
        assert 'danger_zones' in result, "Should have danger zones"
        assert 'ai_instructions' in result, "Should have AI instructions"
        assert 'modification_checklist' in result, "Should have checklist"
        
        # Check danger zones
        danger_zones = result['danger_zones']
        print(f"  ✓ Danger zones summary: {danger_zones.get('summary', 'N/A')}")
        
        # Check for critical files
        do_not_modify = danger_zones.get('do_not_modify', [])
        extreme_caution = danger_zones.get('extreme_caution', [])
        
        print(f"  ✓ DO NOT MODIFY files: {len(do_not_modify)}")
        print(f"  ✓ EXTREME CAUTION files: {len(extreme_caution)}")
        
        # Verify auth files are marked as dangerous
        auth_files = [f for f in do_not_modify + extreme_caution 
                      if 'auth' in f.get('file', '').lower()]
        assert len(auth_files) > 0, "Auth files should be marked as dangerous"
        
        return result
        
    async def test_get_business_context(self):
        """Test retrieving business context"""
        print("\n💼 Testing get_business_context...")
        
        result = await self.server._get_business_context()
        
        assert 'error' not in result, "Should not have errors"
        assert 'domain_model' in result, "Should have domain model"
        assert 'user_journeys' in result, "Should have user journeys"
        assert 'business_rules' in result, "Should have business rules"
        assert 'key_features' in result, "Should have key features"
        
        # Check domain entities
        entities = result['domain_model'].get('entities', {})
        print(f"  ✓ Domain entities found: {len(entities)}")
        assert 'User' in entities, "Should identify User entity"
        
        # Check features
        features = result['key_features']
        print(f"  ✓ Key features: {', '.join(features[:5])}")
        assert 'User Authentication' in features, "Should identify auth feature"
        
        # Check compliance
        compliance = result['compliance_requirements']
        print(f"  ✓ Compliance requirements: {len(compliance)}")
        
        return result
        
    async def test_get_modification_guidance(self):
        """Test modification guidance for specific files"""
        print("\n🛡️ Testing get_modification_guidance...")
        
        # Test general guidance
        general_result = await self.server._get_modification_guidance()
        assert 'error' not in general_result, "Should not have errors"
        assert 'general_guidance' in general_result, "Should have general guidance"
        
        print("  ✓ General guidance retrieved")
        
        # Test specific file guidance - auth service
        auth_file = "src/services/auth_service.py"
        auth_guidance = await self.server._get_modification_guidance(auth_file)
        
        assert 'error' not in auth_guidance, "Should not have errors"
        assert auth_guidance['risk_level'] in ['CRITICAL', 'HIGH'], "Auth service should be high risk"
        assert len(auth_guidance['checklist']) > 0, "Should have checklist items"
        assert len(auth_guidance['safer_alternatives']) > 0, "Should suggest alternatives"
        
        print(f"  ✓ {auth_file}:")
        print(f"    - Risk Level: {auth_guidance['risk_level']}")
        print(f"    - Impact: {auth_guidance['impact_summary']}")
        print(f"    - AI Warning: {auth_guidance['ai_warning'][:100]}...")
        print(f"    - Checklist items: {len(auth_guidance['checklist'])}")
        
        # Test a safer file
        test_file = "tests/test_auth.py"
        test_guidance = await self.server._get_modification_guidance(test_file)
        
        if test_guidance.get('status') != 'not_analyzed':
            print(f"  ✓ {test_file}:")
            print(f"    - Risk Level: {test_guidance.get('risk_level', 'N/A')}")
            print(f"    - Safe to modify: {'Yes' if test_guidance.get('risk_level') == 'LOW' else 'Use caution'}")
            
        return auth_guidance
        
    async def test_full_workflow(self):
        """Test a complete workflow as an AI would use it"""
        print("\n🤖 Testing full AI workflow...")
        
        # Step 1: Analyze the codebase
        print("\n1️⃣ Analyzing codebase...")
        analysis = await self.test_analyze_codebase()
        
        # Step 2: Get AI knowledge package
        print("\n2️⃣ Getting AI knowledge package...")
        ai_package = await self.test_get_ai_knowledge_package()
        
        # Step 3: Understand business context
        print("\n3️⃣ Understanding business context...")
        business = await self.test_get_business_context()
        
        # Step 4: Check modification guidance for critical file
        print("\n4️⃣ Checking modification guidance...")
        guidance = await self.test_get_modification_guidance()
        
        # Simulate AI reading the instant context
        print("\n📖 AI Reading Instant Context:")
        print("-" * 60)
        print(ai_package['instant_context'])
        print("-" * 60)
        
        # Simulate AI checking danger zones
        print("\n⚠️ AI Checking Danger Zones:")
        danger_zones = ai_package['danger_zones']
        if danger_zones.get('do_not_modify'):
            print("  DO NOT MODIFY:")
            for item in danger_zones['do_not_modify'][:3]:
                print(f"    - {item['file']}: {item['reason']}")
                
        # Simulate AI understanding the business
        print("\n💼 AI Understanding Business:")
        print(f"  Executive Summary: {business.get('executive_summary', 'N/A')[:200]}...")
        
        print("\n✅ Full workflow completed successfully!")
        print("🎯 The enhanced CodebaseIQ Pro provides immediate, comprehensive understanding")
        print("🛡️ AI assistants can now make safe, informed modifications from the first message")
        
    async def run_all_tests(self):
        """Run all integration tests"""
        try:
            # Initialize server
            print("🚀 Initializing CodebaseIQ Pro Server...")
            self.server = CodebaseIQProServer()
            
            # Create test codebase
            self.create_test_codebase()
            
            # Run full workflow test
            await self.test_full_workflow()
            
            print("\n✨ All integration tests passed!")
            print("📊 Enhanced CodebaseIQ Pro is working correctly")
            print("🎯 AI assistants now have 100% useful context at conversation startup")
            
        finally:
            # Cleanup
            if self.test_dir and os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                print(f"\n🧹 Cleaned up test directory")

async def main():
    """Run the integration tests"""
    tester = TestCodebaseIQIntegration()
    await tester.run_all_tests()

if __name__ == "__main__":
    print("🧪 CodebaseIQ Pro Integration Test Suite")
    print("=" * 60)
    asyncio.run(main())