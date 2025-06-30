#!/usr/bin/env python3
"""Test the Deep Understanding Agent - comprehensive validation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from codebaseiq.agents.deep_understanding_agent import DeepUnderstandingAgent, CodeContext
import json

def test_deep_understanding():
    """Test that the agent can extract semantic meaning from various code patterns"""
    
    agent = DeepUnderstandingAgent()
    
    # Test 1: Authentication module (security-critical)
    auth_code = '''
"""User authentication module for the web application
Handles login, logout, and session management
"""

import bcrypt
import jwt
from datetime import datetime, timedelta
from database import db
from models import User
from utils.rate_limiter import check_rate_limit

class AuthenticationService:
    """Handles user login and session management with security best practices"""
    
    def __init__(self):
        self.db = db
        self.secret_key = os.environ.get('JWT_SECRET')
        
    async def login(self, email: str, password: str, ip_address: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user and create a JWT session.
        
        Args:
            email: User's email address
            password: Plain text password
            ip_address: Client IP for rate limiting
            
        Returns:
            Dict with user data and JWT token if successful, None otherwise
            
        Raises:
            RateLimitException: If too many login attempts
        """
        # Check rate limiting
        if not check_rate_limit(ip_address, 'login', max_attempts=5):
            logger.warning(f"Rate limit exceeded for IP: {ip_address}")
            raise RateLimitException("Too many login attempts")
            
        try:
            # BEGIN TRANSACTION
            user = await self.db.query(User).filter_by(email=email).first()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash):
                # Generate JWT token
                token = self._generate_jwt(user)
                
                # Update last login
                user.last_login = datetime.utcnow()
                await self.db.commit()
                
                # Log successful login for security audit
                logger.info(f"Successful login for user {email} from IP {ip_address}")
                
                return {
                    'user_id': user.id,
                    'email': user.email,
                    'token': token,
                    'expires_in': 3600
                }
            else:
                # Log failed attempt
                logger.warning(f"Failed login attempt for {email} from IP {ip_address}")
                return None
                
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Login error: {e}")
            raise
            
    def _generate_jwt(self, user: User) -> str:
        """Generate a secure JWT token"""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    '''
    
    # Analyze the authentication code
    print("\n=== Test 1: Authentication Module ===")
    auth_context = agent.analyze_file("auth.py", auth_code)
    
    print(f"✅ Purpose: {auth_context.purpose}")
    print(f"✅ Business Logic: {auth_context.business_logic}")
    print(f"✅ Modification Risk: {auth_context.modification_risk}")
    print(f"✅ Security Concerns: {auth_context.security_concerns}")
    print(f"✅ AI Guidance: {auth_context.ai_guidance}")
    print(f"✅ Critical Functions: {len(auth_context.critical_functions)}")
    
    # Validate critical aspects
    assert "authentication" in auth_context.purpose.lower()
    assert "CRITICAL" in auth_context.modification_risk or "HIGH" in auth_context.modification_risk
    assert len(auth_context.security_concerns) > 3  # Should detect multiple security patterns
    assert any("login" in func['name'] for func in auth_context.critical_functions)
    
    # Test 2: API Endpoint Handler
    api_code = '''
"""Product API endpoints for the e-commerce platform"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from models import Product, ProductCreate, ProductUpdate
from database import get_db
from auth import get_current_user
from cache import cache_key, invalidate_cache

router = APIRouter(prefix="/api/v1/products", tags=["products"])

@router.get("/", response_model=List[Product])
@cache_key("products:list", ttl=300)
async def get_products(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    db = Depends(get_db)
):
    """
    Get list of products with pagination and filtering.
    Results are cached for 5 minutes.
    """
    query = db.query(Product)
    
    if category:
        query = query.filter(Product.category == category)
        
    products = query.offset(skip).limit(limit).all()
    return products

@router.post("/", response_model=Product)
async def create_product(
    product: ProductCreate,
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Create a new product. Requires admin authentication.
    Invalidates the product list cache.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    db_product = Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    
    # Invalidate cache
    await invalidate_cache("products:list")
    
    return db_product

@router.delete("/{product_id}")
async def delete_product(
    product_id: int,
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Delete a product. Requires admin authentication."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
        
    db.delete(product)
    db.commit()
    
    # Invalidate cache
    await invalidate_cache("products:list")
    
    return {"status": "deleted"}
    '''
    
    print("\n=== Test 2: API Endpoints ===")
    api_context = agent.analyze_file("api/products.py", api_code)
    
    print(f"✅ Purpose: {api_context.purpose}")
    print(f"✅ Business Logic: {api_context.business_logic}")
    print(f"✅ Side Effects: {api_context.side_effects}")
    print(f"✅ Dependencies: {len(api_context.dependencies)}")
    
    assert "API endpoint" in api_context.purpose or "endpoint" in api_context.business_logic
    assert "Database operations" in api_context.side_effects
    assert "Cache operations" in api_context.side_effects
    
    # Test 3: React Component (JavaScript)
    react_code = '''
/**
 * UserDashboard Component
 * Main dashboard view for authenticated users showing analytics and recent activity
 */
 
import React, { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import { api } from '../services/api';
import { LineChart, BarChart } from '../components/Charts';
import { ActivityFeed } from '../components/ActivityFeed';
import { ErrorBoundary } from '../components/ErrorBoundary';

export const UserDashboard = () => {
  const { user } = useAuth();
  const [analytics, setAnalytics] = useState(null);
  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch user analytics and activities in parallel
        const [analyticsRes, activitiesRes] = await Promise.all([
          api.get(`/users/${user.id}/analytics`),
          api.get(`/users/${user.id}/activities?limit=10`)
        ]);
        
        setAnalytics(analyticsRes.data);
        setActivities(activitiesRes.data);
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
        setError('Unable to load dashboard data. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    if (user?.id) {
      fetchDashboardData();
    }
  }, [user]);
  
  const handleExportData = async () => {
    try {
      const response = await api.get(`/users/${user.id}/export`, {
        responseType: 'blob'
      });
      
      // Download the file
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `user-data-${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
    } catch (err) {
      console.error('Export failed:', err);
    }
  };
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div className="error">{error}</div>;
  
  return (
    <ErrorBoundary>
      <div className="dashboard-container">
        <h1>Welcome back, {user.name}!</h1>
        
        <div className="analytics-section">
          <LineChart data={analytics?.revenue} title="Revenue Trends" />
          <BarChart data={analytics?.products} title="Product Performance" />
        </div>
        
        <ActivityFeed activities={activities} />
        
        <button onClick={handleExportData}>Export My Data</button>
      </div>
    </ErrorBoundary>
  );
};
    '''
    
    print("\n=== Test 3: React Component ===")
    react_context = agent.analyze_file("components/UserDashboard.jsx", react_code)
    
    print(f"✅ Purpose: {react_context.purpose}")
    print(f"✅ Language: {react_context.language}")
    print(f"✅ Business Logic: {react_context.business_logic}")
    print(f"✅ Critical Functions: {[f['name'] for f in react_context.critical_functions[:3]]}")
    
    assert react_context.language == "javascript"
    assert "React" in react_context.purpose or "UI" in react_context.purpose
    assert any("export" in func['name'].lower() for func in react_context.critical_functions)
    
    # Test 4: Database Migration (High Risk)
    migration_code = '''
"""
Database migration: Add user roles and permissions
CRITICAL: This migration affects authentication and authorization
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '3f2e4d5c6b7a'
down_revision = '1a2b3c4d5e6f'

def upgrade():
    """Add roles and permissions tables for RBAC"""
    
    # Create roles table
    op.create_table('roles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(50), nullable=False, unique=True),
        sa.Column('description', sa.String(200)),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create permissions table
    op.create_table('permissions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('resource', sa.String(50), nullable=False),
        sa.Column('action', sa.String(50), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create role_permissions junction table
    op.create_table('role_permissions',
        sa.Column('role_id', sa.Integer(), nullable=False),
        sa.Column('permission_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['permission_id'], ['permissions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('role_id', 'permission_id')
    )
    
    # Add role_id to users table
    op.add_column('users', sa.Column('role_id', sa.Integer()))
    op.create_foreign_key('fk_users_role', 'users', 'roles', ['role_id'], ['id'])
    
    # Insert default roles
    op.execute("""
        INSERT INTO roles (name, description, created_at) VALUES
        ('admin', 'Full system access', NOW()),
        ('user', 'Regular user access', NOW()),
        ('guest', 'Limited read-only access', NOW())
    """)
    
def downgrade():
    """Remove roles and permissions - DANGER: This will break authentication!"""
    op.drop_constraint('fk_users_role', 'users', type_='foreignkey')
    op.drop_column('users', 'role_id')
    op.drop_table('role_permissions')
    op.drop_table('permissions')
    op.drop_table('roles')
    '''
    
    print("\n=== Test 4: Database Migration ===")
    migration_context = agent.analyze_file("migrations/add_rbac.py", migration_code)
    
    print(f"✅ Purpose: {migration_context.purpose}")
    print(f"✅ Modification Risk: {migration_context.modification_risk}")
    print(f"✅ AI Guidance: {migration_context.ai_guidance}")
    
    assert "CRITICAL" in migration_context.modification_risk
    assert "Database schema changes" in migration_context.modification_risk
    
    # Test 5: Configuration File (Non-code)
    config_code = '''
# Production configuration for the application
# WARNING: Changes to this file affect the live system!

[database]
host = "prod-db.example.com"
port = 5432
name = "production_db"
pool_size = 20
max_overflow = 40

[redis]
host = "prod-redis.example.com"
port = 6379
db = 0
password = "${REDIS_PASSWORD}"

[api]
rate_limit = 1000  # requests per minute
timeout = 30  # seconds
max_payload_size = "10MB"

[security]
jwt_expiry = 3600  # 1 hour
refresh_token_expiry = 604800  # 7 days
password_min_length = 12
require_2fa = true

[features]
enable_analytics = true
enable_exports = true
maintenance_mode = false
    '''
    
    print("\n=== Test 5: Configuration File ===")
    config_context = agent.analyze_file("config/production.ini", config_code)
    
    print(f"✅ Purpose: {config_context.purpose}")
    print(f"✅ Modification Risk: {config_context.modification_risk}")
    
    # Generate and display the understanding summary
    print("\n=== Understanding Summary ===")
    summary = agent.generate_understanding_summary()
    
    print(f"Total files analyzed: {summary['total_files_analyzed']}")
    print(f"Languages found: {summary['languages_found']}")
    print(f"Critical files: {len(summary['critical_files'])}")
    print(f"Security sensitive files: {len(summary['security_sensitive_files'])}")
    print("\nImmediate AI Guidance:")
    print(summary['immediate_ai_guidance'])
    
    # Final validation
    assert summary['total_files_analyzed'] == 5
    assert 'python' in summary['languages_found']
    assert 'javascript' in summary['languages_found']
    assert len(summary['critical_files']) >= 2  # auth.py and migration should be critical
    assert len(summary['security_sensitive_files']) >= 1  # auth.py at minimum
    
    print("\n✅ All Deep Understanding Agent tests passed!")
    print("✅ The agent successfully extracts semantic meaning and provides immediate AI guidance")
    
    # Save a sample context for inspection
    sample_output = {
        "auth_context": auth_context.to_dict(),
        "summary": summary
    }
    
    with open("tests/test_deep_understanding_output.json", "w") as f:
        json.dump(sample_output, f, indent=2)
        print("\n📄 Sample output saved to tests/test_deep_understanding_output.json")

if __name__ == "__main__":
    test_deep_understanding()