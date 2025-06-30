#!/usr/bin/env python3
"""Test Business Logic Extractor - domain understanding and business rules"""

import sys
import os
import json

# Direct imports to avoid dependency issues
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import modules directly
import importlib.util

# Load required modules
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = os.path.dirname(os.path.abspath(__file__))
deep_module = load_module("deep_understanding_agent", 
    os.path.join(base_path, "src/codebaseiq/agents/deep_understanding_agent.py"))
cross_module = load_module("cross_file_intelligence",
    os.path.join(base_path, "src/codebaseiq/agents/cross_file_intelligence.py"))
business_module = load_module("business_logic_extractor",
    os.path.join(base_path, "src/codebaseiq/agents/business_logic_extractor.py"))

DeepUnderstandingAgent = deep_module.DeepUnderstandingAgent
CrossFileIntelligence = cross_module.CrossFileIntelligence
BusinessLogicExtractor = business_module.BusinessLogicExtractor

def test_business_logic_extractor():
    """Test business logic extraction and domain understanding"""
    
    print("💼 Testing Business Logic Extractor...")
    
    # Create test codebase with business logic
    file_contents = {
        "models/user.py": '''
"""User model for the e-commerce platform"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    """Represents a customer or admin user in the system"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    orders = relationship("Order", back_populates="user")
    cart = relationship("Cart", uselist=False, back_populates="user")
    addresses = relationship("Address", back_populates="user")
    payment_methods = relationship("PaymentMethod", back_populates="user")
    
    def has_permission(self, permission):
        """Check if user has a specific permission"""
        if self.is_admin:
            return True
        # RULE: Regular users can only access their own data
        return permission in ['read_own', 'update_own']
        
    def can_purchase(self):
        """Business rule: Check if user can make purchases"""
        # REQUIREMENT: Users must be active and have verified email
        return self.is_active and self.email_verified
''',

        "models/order.py": '''
"""Order model for tracking customer purchases"""
from sqlalchemy import Column, Integer, String, DateTime, Numeric, ForeignKey, Enum
from sqlalchemy.orm import relationship
from database import Base
import enum

class OrderStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class Order(Base):
    """Represents a customer order/transaction"""
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    order_number = Column(String(20), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING)
    total_amount = Column(Numeric(10, 2), nullable=False)
    tax_amount = Column(Numeric(10, 2), default=0)
    shipping_amount = Column(Numeric(10, 2), default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    shipped_at = Column(DateTime)
    delivered_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")
    payment = relationship("Payment", uselist=False, back_populates="order")
    
    def can_cancel(self):
        """Business rule: Orders can only be cancelled if not shipped"""
        return self.status in [OrderStatus.PENDING, OrderStatus.PROCESSING]
        
    def calculate_total(self):
        """Calculate order total with business rules"""
        subtotal = sum(item.price * item.quantity for item in self.items)
        
        # RULE: Free shipping for orders over $100
        if subtotal > 100:
            self.shipping_amount = 0
        else:
            self.shipping_amount = 10
            
        # RULE: Tax is 8% of subtotal
        self.tax_amount = subtotal * 0.08
        
        self.total_amount = subtotal + self.tax_amount + self.shipping_amount
        return self.total_amount
''',

        "services/order_service.py": '''
"""Order processing service with business logic"""
from models import Order, OrderItem, Product, User
from payment_service import PaymentService
from notification_service import NotificationService
from inventory_service import InventoryService

class OrderService:
    """Handles order creation and processing workflow"""
    
    def __init__(self):
        self.payment_service = PaymentService()
        self.notification_service = NotificationService()
        self.inventory_service = InventoryService()
        
    def create_order(self, user: User, cart_items: list) -> Order:
        """
        Create a new order from cart items.
        
        Business Process:
        1. Validate user can purchase
        2. Check inventory availability
        3. Calculate pricing
        4. Create order
        5. Reserve inventory
        """
        # RULE: User must be eligible to purchase
        if not user.can_purchase():
            raise ValueError("User is not eligible to make purchases")
            
        # RULE: Minimum order amount is $10
        subtotal = sum(item['price'] * item['quantity'] for item in cart_items)
        if subtotal < 10:
            raise ValueError("Minimum order amount is $10")
            
        # Check inventory for all items
        for item in cart_items:
            if not self.inventory_service.check_availability(item['product_id'], item['quantity']):
                raise ValueError(f"Product {item['product_id']} is out of stock")
                
        # Create order
        order = Order(user_id=user.id, status=OrderStatus.PENDING)
        
        # Add items to order
        for item in cart_items:
            order_item = OrderItem(
                order=order,
                product_id=item['product_id'],
                quantity=item['quantity'],
                price=item['price']
            )
            order.items.append(order_item)
            
        # Calculate totals
        order.calculate_total()
        
        # Reserve inventory
        for item in order.items:
            self.inventory_service.reserve(item.product_id, item.quantity)
            
        return order
        
    def process_payment(self, order: Order, payment_method: str) -> bool:
        """
        Process payment for an order.
        
        COMPLIANCE: PCI DSS - Never store credit card details
        """
        # RULE: Payment must be processed within 30 minutes of order creation
        if (datetime.utcnow() - order.created_at).seconds > 1800:
            raise ValueError("Order has expired. Please create a new order.")
            
        # Process payment
        payment_result = self.payment_service.charge(
            amount=order.total_amount,
            payment_method=payment_method,
            order_id=order.id
        )
        
        if payment_result.success:
            order.status = OrderStatus.PROCESSING
            # Send confirmation email
            self.notification_service.send_order_confirmation(order)
            # Trigger fulfillment
            self._trigger_fulfillment(order)
            return True
        else:
            # Release reserved inventory
            for item in order.items:
                self.inventory_service.release(item.product_id, item.quantity)
            order.status = OrderStatus.CANCELLED
            return False
            
    def _trigger_fulfillment(self, order: Order):
        """Internal method to trigger order fulfillment"""
        # Send to warehouse system
        pass
''',

        "api/user_auth.py": '''
"""User authentication API endpoints"""
from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash
from models import User
from utils.jwt_helper import generate_token, verify_token
import re

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    User registration endpoint.
    
    Business Rules:
    - Email must be valid format
    - Password minimum 8 characters
    - Username must be unique
    """
    data = request.get_json()
    
    # RULE: Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, data['email']):
        return jsonify({"error": "Invalid email format"}), 400
        
    # RULE: Password must be at least 8 characters
    if len(data['password']) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
        
    # RULE: Username must be alphanumeric and 3-20 characters
    if not re.match(r'^[a-zA-Z0-9]{3,20}$', data['username']):
        return jsonify({"error": "Username must be alphanumeric and 3-20 characters"}), 400
        
    # Check if user exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already registered"}), 409
        
    # Create new user
    user = User(
        email=data['email'],
        username=data['username'],
        password_hash=generate_password_hash(data['password'])
    )
    
    db.session.add(user)
    db.session.commit()
    
    # Send welcome email
    send_welcome_email(user.email)
    
    return jsonify({
        "message": "Registration successful",
        "user_id": user.id
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint with rate limiting"""
    data = request.get_json()
    
    # SECURITY: Rate limiting - max 5 attempts per 15 minutes
    if not check_rate_limit(request.remote_addr, 'login'):
        return jsonify({"error": "Too many login attempts"}), 429
        
    user = User.query.filter_by(email=data['email']).first()
    
    if user and check_password_hash(user.password_hash, data['password']):
        # RULE: Inactive users cannot login
        if not user.is_active:
            return jsonify({"error": "Account is deactivated"}), 403
            
        # Generate JWT token
        token = generate_token(user.id)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            "token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "is_admin": user.is_admin
            }
        }), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401
''',

        "services/subscription_service.py": '''
"""Subscription management service"""
from models import User, Subscription, Plan
from datetime import datetime, timedelta
from payment_service import PaymentService

class SubscriptionService:
    """Manages user subscriptions and recurring billing"""
    
    # BUSINESS RULE: Trial period is 14 days
    TRIAL_PERIOD_DAYS = 14
    
    # BUSINESS RULE: Grace period for failed payments
    GRACE_PERIOD_DAYS = 3
    
    def create_subscription(self, user: User, plan_id: str, payment_method: str = None):
        """Create a new subscription for a user"""
        
        plan = Plan.query.get(plan_id)
        if not plan:
            raise ValueError("Invalid plan")
            
        # RULE: Users can only have one active subscription
        if user.active_subscription:
            raise ValueError("User already has an active subscription")
            
        # RULE: Premium plans require payment method
        if plan.price > 0 and not payment_method:
            raise ValueError("Payment method required for paid plans")
            
        subscription = Subscription(
            user_id=user.id,
            plan_id=plan_id,
            status='trial' if plan.has_trial else 'active',
            trial_ends_at=datetime.utcnow() + timedelta(days=self.TRIAL_PERIOD_DAYS) if plan.has_trial else None,
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=30)
        )
        
        # Process initial payment for paid plans without trial
        if plan.price > 0 and not plan.has_trial:
            if not self._process_subscription_payment(subscription, payment_method):
                raise ValueError("Payment failed")
                
        db.session.add(subscription)
        db.session.commit()
        
        return subscription
        
    def check_subscription_status(self, user: User) -> dict:
        """Check user's subscription status and features"""
        
        subscription = user.active_subscription
        if not subscription:
            return {"status": "none", "features": []}
            
        # RULE: Check trial expiration
        if subscription.status == 'trial' and subscription.trial_ends_at < datetime.utcnow():
            subscription.status = 'expired'
            db.session.commit()
            
        # RULE: Check payment due
        if subscription.status == 'active' and subscription.current_period_end < datetime.utcnow():
            if not self._process_recurring_payment(subscription):
                subscription.status = 'past_due'
                subscription.grace_period_ends = datetime.utcnow() + timedelta(days=self.GRACE_PERIOD_DAYS)
                db.session.commit()
                
        return {
            "status": subscription.status,
            "plan": subscription.plan.name,
            "features": subscription.plan.features,
            "next_billing_date": subscription.current_period_end
        }
'''
    }
    
    # Phase 1: Deep Understanding
    print("\n📊 Phase 1: Deep Understanding Analysis")
    deep_agent = DeepUnderstandingAgent()
    contexts = {}
    
    for file_path, content in file_contents.items():
        context = deep_agent.analyze_file(file_path, content)
        contexts[file_path] = context
        print(f"  ✓ Analyzed: {file_path}")
    
    # Phase 2: Cross-File Intelligence
    print("\n🔗 Phase 2: Cross-File Intelligence")
    cross_intel = CrossFileIntelligence()
    cross_results = cross_intel.analyze_relationships(contexts, file_contents)
    print("  ✓ Relationships mapped")
    
    # Phase 3: Business Logic Extraction
    print("\n💼 Phase 3: Business Logic Extraction")
    business_extractor = BusinessLogicExtractor()
    business_results = business_extractor.extract_business_logic(contexts, cross_results, file_contents)
    
    # Display results
    print("\n📈 Domain Model:")
    if business_results['domain_model']['entities']:
        for entity_name, entity_data in list(business_results['domain_model']['entities'].items())[:5]:
            print(f"  {entity_name}:")
            print(f"    Purpose: {entity_data['business_purpose']}")
            print(f"    Importance: {entity_data['importance']}")
            print(f"    Operations: {entity_data['crud_operations']}")
    
    print("\n🚶 User Journeys:")
    for journey in business_results['user_journeys'][:3]:
        print(f"  {journey['journey_type'].replace('_', ' ').title()}:")
        print(f"    Description: {journey['description']}")
        print(f"    Complexity: {journey['complexity']}")
        print(f"    Files involved: {len(journey['involved_files'])}")
        if journey['critical_points']:
            print(f"    Critical points: {journey['critical_points'][0]}")
    
    print("\n📋 Business Rules (Top 5):")
    for rule in business_results['business_rules'][:5]:
        print(f"  Rule: {rule['rule']}")
        print(f"    Type: {rule['type']}")
        print(f"    Category: {rule['category']}")
        print(f"    Impact: {rule['business_impact']}")
        print(f"    File: {rule['file']}")
    
    print("\n🌟 Key Features:")
    for feature in business_results['key_features'][:10]:
        print(f"  - {feature}")
    
    print("\n🔒 Compliance Requirements:")
    for compliance in business_results['compliance_requirements']:
        print(f"  - {compliance}")
    
    print("\n📝 Executive Summary:")
    print(business_results['executive_summary'])
    
    print("\n🤖 AI Business Context:")
    print(business_results['ai_business_context'])
    
    print("\n💡 Immediate Context:")
    print(business_results['immediate_context'])
    
    # Validate results
    assert len(business_results['domain_model']['entities']) > 0, "Should extract domain entities"
    assert 'User' in business_results['domain_model']['entities'], "Should identify User entity"
    assert 'Order' in business_results['domain_model']['entities'], "Should identify Order entity"
    assert len(business_results['user_journeys']) > 0, "Should identify user journeys"
    assert len(business_results['business_rules']) > 0, "Should extract business rules"
    assert any('minimum' in rule['rule'].lower() for rule in business_results['business_rules']), "Should find minimum order rule"
    assert len(business_results['key_features']) > 0, "Should identify key features"
    assert 'Payment Processing' in business_results['key_features'], "Should identify payment feature"
    assert len(business_results['compliance_requirements']) > 0, "Should identify compliance requirements"
    
    print("\n✅ Business Logic Extractor Test Passed!")
    print("✅ Successfully extracts domain model, user journeys, and business rules")
    print("✅ Provides immediate business context for AI understanding")
    
    # Save results
    output = {
        "domain_entities": list(business_results['domain_model']['entities'].keys()),
        "user_journeys": [j['journey_type'] for j in business_results['user_journeys']],
        "business_rule_categories": list(set(r['category'] for r in business_results['business_rules'])),
        "key_features": business_results['key_features'][:10],
        "compliance": list(business_results['compliance_requirements']),
        "immediate_context": business_results['immediate_context']
    }
    
    with open("test_business_logic_output.json", "w") as f:
        json.dump(output, f, indent=2)
        print("\n📄 Results saved to test_business_logic_output.json")

if __name__ == "__main__":
    test_business_logic_extractor()