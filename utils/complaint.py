"""
Complaint Management System for CitizenAI
Handles complaint submission, tracking, and resolution workflow
"""

import streamlit as st
import sqlite3
import uuid
import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

class ComplaintManager:
    """Manages citizen complaints and government responses"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.init_complaint_tables()
    
    def init_complaint_tables(self):
        """Initialize complaint-related database tables"""
        c = self.db.cursor()
        
        # Complaints table
        c.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id TEXT PRIMARY KEY,
            citizen_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT DEFAULT 'Medium',
            status TEXT DEFAULT 'Open',
            location TEXT,
            photo_path TEXT,
            assigned_to TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (citizen_id) REFERENCES users(id),
            FOREIGN KEY (assigned_to) REFERENCES users(id)
        )
        ''')
        
        # Complaint updates/comments table
        c.execute('''
        CREATE TABLE IF NOT EXISTS complaint_updates (
            id TEXT PRIMARY KEY,
            complaint_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            update_text TEXT NOT NULL,
            update_type TEXT DEFAULT 'comment',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (complaint_id) REFERENCES complaints(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        # Complaint categories reference
        c.execute('''
        CREATE TABLE IF NOT EXISTS complaint_categories (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            estimated_resolution_days INTEGER DEFAULT 7,
            auto_assign_to TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Insert default categories if not exist
        default_categories = [
            ('road-maintenance', 'Road Maintenance', 'Public Works', 14, None),
            ('water-sewage', 'Water & Sewage', 'Public Utilities', 3, None),
            ('garbage-collection', 'Garbage Collection', 'Sanitation', 2, None),
            ('street-lighting', 'Street Lighting', 'Public Works', 7, None),
            ('noise-complaint', 'Noise Complaint', 'Public Safety', 1, None),
            ('building-permits', 'Building Permits', 'Planning Department', 30, None),
            ('park-maintenance', 'Park Maintenance', 'Parks & Recreation', 10, None),
            ('traffic-issues', 'Traffic Issues', 'Transportation', 5, None),
            ('other', 'Other', 'General Services', 7, None)
        ]
        
        for cat_id, name, dept, days, auto_assign in default_categories:
            c.execute('''
                INSERT OR IGNORE INTO complaint_categories 
                (id, name, department, estimated_resolution_days, auto_assign_to) 
                VALUES (?, ?, ?, ?, ?)
            ''', (cat_id, name, dept, days, auto_assign))
        
        self.db.commit()
    
    def submit_complaint(self, citizen_id: str, title: str, description: str, 
                        category: str, location: str = None, priority: str = 'Medium') -> Tuple[bool, str]:
        """Submit a new complaint"""
        try:
            c = self.db.cursor()
            complaint_id = str(uuid.uuid4())
            
            # Auto-assign based on category if available
            c.execute("SELECT auto_assign_to FROM complaint_categories WHERE id = ?", (category,))
            result = c.fetchone()
            assigned_to = result[0] if result and result[0] else None
            
            c.execute('''
                INSERT INTO complaints 
                (id, citizen_id, title, description, category, priority, location, assigned_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (complaint_id, citizen_id, title, description, category, priority, location, assigned_to))
            
            # Add initial update
            update_id = str(uuid.uuid4())
            c.execute('''
                INSERT INTO complaint_updates 
                (id, complaint_id, user_id, update_text, update_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (update_id, complaint_id, citizen_id, "Complaint submitted", "status_change"))
            
            self.db.commit()
            return True, complaint_id
            
        except Exception as e:
            self.db.rollback()
            return False, str(e)
    
    def get_complaints_by_citizen(self, citizen_id: str) -> List[Dict]:
        """Get all complaints submitted by a citizen"""
        c = self.db.cursor()
        c.execute('''
            SELECT c.*, cc.name as category_name, cc.department,
                   u.username as assigned_to_name
            FROM complaints c
            LEFT JOIN complaint_categories cc ON c.category = cc.id
            LEFT JOIN users u ON c.assigned_to = u.id
            WHERE c.citizen_id = ?
            ORDER BY c.created_at DESC
        ''', (citizen_id,))
        
        columns = [desc[0] for desc in c.description]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    
    def get_all_complaints(self, status_filter: str = None, assigned_to: str = None) -> List[Dict]:
        """Get all complaints with optional filters"""
        c = self.db.cursor()
        
        query = '''
            SELECT c.*, cc.name as category_name, cc.department,
                   u1.username as citizen_name, u2.username as assigned_to_name
            FROM complaints c
            LEFT JOIN complaint_categories cc ON c.category = cc.id
            LEFT JOIN users u1 ON c.citizen_id = u1.id
            LEFT JOIN users u2 ON c.assigned_to = u2.id
            WHERE 1=1
        '''
        params = []
        
        if status_filter:
            query += " AND c.status = ?"
            params.append(status_filter)
        
        if assigned_to:
            query += " AND c.assigned_to = ?"
            params.append(assigned_to)
        
        query += " ORDER BY c.created_at DESC"
        
        c.execute(query, params)
        columns = [desc[0] for desc in c.description]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    
    def update_complaint_status(self, complaint_id: str, new_status: str, 
                               user_id: str, comment: str = None) -> bool:
        """Update complaint status"""
        try:
            c = self.db.cursor()
            
            # Update complaint
            update_time = datetime.datetime.now()
            resolved_time = update_time if new_status == 'Resolved' else None
            
            c.execute('''
                UPDATE complaints 
                SET status = ?, updated_at = ?, resolved_at = ?
                WHERE id = ?
            ''', (new_status, update_time, resolved_time, complaint_id))
            
            # Add update record
            update_id = str(uuid.uuid4())
            update_text = f"Status changed to: {new_status}"
            if comment:
                update_text += f" - {comment}"
            
            c.execute('''
                INSERT INTO complaint_updates 
                (id, complaint_id, user_id, update_text, update_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (update_id, complaint_id, user_id, update_text, "status_change"))
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            return False
    
    def assign_complaint(self, complaint_id: str, assigned_to: str, assigner_id: str) -> bool:
        """Assign complaint to a government official"""
        try:
            c = self.db.cursor()
            
            c.execute('''
                UPDATE complaints 
                SET assigned_to = ?, updated_at = ?
                WHERE id = ?
            ''', (assigned_to, datetime.datetime.now(), complaint_id))
            
            # Get assignee name
            c.execute("SELECT username FROM users WHERE id = ?", (assigned_to,))
            assignee_name = c.fetchone()[0]
            
            # Add update record
            update_id = str(uuid.uuid4())
            c.execute('''
                INSERT INTO complaint_updates 
                (id, complaint_id, user_id, update_text, update_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (update_id, complaint_id, assigner_id, f"Assigned to: {assignee_name}", "assignment"))
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            return False
    
    def add_complaint_comment(self, complaint_id: str, user_id: str, comment: str) -> bool:
        """Add a comment to a complaint"""
        try:
            c = self.db.cursor()
            update_id = str(uuid.uuid4())
            
            c.execute('''
                INSERT INTO complaint_updates 
                (id, complaint_id, user_id, update_text, update_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (update_id, complaint_id, user_id, comment, "comment"))
            
            # Update complaint's updated_at timestamp
            c.execute('''
                UPDATE complaints 
                SET updated_at = ?
                WHERE id = ?
            ''', (datetime.datetime.now(), complaint_id))
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            return False
    
    def get_complaint_updates(self, complaint_id: str) -> List[Dict]:
        """Get all updates for a complaint"""
        c = self.db.cursor()
        c.execute('''
            SELECT cu.*, u.username, u.user_type
            FROM complaint_updates cu
            JOIN users u ON cu.user_id = u.id
            WHERE cu.complaint_id = ?
            ORDER BY cu.created_at ASC
        ''', (complaint_id,))
        
        columns = [desc[0] for desc in c.description]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    
    def get_complaint_categories(self) -> List[Dict]:
        """Get all complaint categories"""
        c = self.db.cursor()
        c.execute("SELECT * FROM complaint_categories ORDER BY name")
        columns = [desc[0] for desc in c.description]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    
    def get_complaint_statistics(self) -> Dict:
        """Get complaint statistics for dashboard"""
        c = self.db.cursor()
        
        stats = {}
        
        # Total counts by status
        c.execute('''
            SELECT status, COUNT(*) 
            FROM complaints 
            GROUP BY status
        ''')
        stats['by_status'] = dict(c.fetchall())
        
        # Total counts by category
        c.execute('''
            SELECT cc.name, COUNT(c.id) 
            FROM complaint_categories cc
            LEFT JOIN complaints c ON cc.id = c.category
            GROUP BY cc.id, cc.name
        ''')
        stats['by_category'] = dict(c.fetchall())
        
        # Monthly trends
        c.execute('''
            SELECT DATE(created_at) as date, COUNT(*) 
            FROM complaints 
            WHERE created_at >= date('now', '-30 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''')
        stats['monthly_trends'] = dict(c.fetchall())
        
        # Average resolution time
        c.execute('''
            SELECT AVG(
                julianday(resolved_at) - julianday(created_at)
            ) as avg_days
            FROM complaints 
            WHERE status = 'Resolved' AND resolved_at IS NOT NULL
        ''')
        result = c.fetchone()
        stats['avg_resolution_days'] = round(result[0], 1) if result[0] else 0
        
        return stats


def render_complaint_submission_form(complaint_manager: ComplaintManager, user_id: str):
    """Render the complaint submission form"""
    st.subheader("📝 Submit New Complaint")
    
    with st.form("complaint_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Complaint Title*", placeholder="Brief description of the issue")
            
            categories = complaint_manager.get_complaint_categories()
            category_options = {cat['name']: cat['id'] for cat in categories}
            selected_category = st.selectbox("Category*", list(category_options.keys()))
            
            priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)
        
        with col2:
            location = st.text_input("Location", placeholder="Street address or landmark")
            
            # Photo upload (placeholder for now)
            photo = st.file_uploader("Upload Photo (Optional)", type=['jpg', 'jpeg', 'png'])
        
        description = st.text_area("Detailed Description*", 
                                 placeholder="Please provide detailed information about the issue...",
                                 height=150)
        
        submitted = st.form_submit_button("Submit Complaint", use_container_width=True)
        
        if submitted:
            if not title or not description:
                st.error("Please fill in all required fields (Title and Description)")
            else:
                category_id = category_options[selected_category]
                success, result = complaint_manager.submit_complaint(
                    user_id, title, description, category_id, location, priority
                )
                
                if success:
                    st.success(f"Complaint submitted successfully! Tracking ID: {result[:8]}")
                    st.info("You can track your complaint status in the 'My Complaints' section.")
                    st.rerun()
                else:
                    st.error(f"Failed to submit complaint: {result}")


def render_citizen_complaints_view(complaint_manager: ComplaintManager, user_id: str):
    """Render citizen's complaint tracking view"""
    st.subheader("📋 My Complaints")
    
    complaints = complaint_manager.get_complaints_by_citizen(user_id)
    
    if not complaints:
        st.info("You haven't submitted any complaints yet.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by Status", 
                                   ["All", "Open", "In Progress", "Resolved", "Closed"])
    
    # Apply filters
    if status_filter != "All":
        complaints = [c for c in complaints if c['status'] == status_filter]
    
    # Display complaints
    for complaint in complaints:
        with st.expander(f"#{complaint['id'][:8]} - {complaint['title']}", 
                        expanded=False):
            
            col1, col2, col3 = st.columns(3)
            with col1:
                status_color = {"Open": "🔵", "In Progress": "🟡", "Resolved": "🟢", "Closed": "⚫"}
                st.write(f"**Status:** {status_color.get(complaint['status'], '⚪')} {complaint['status']}")
            with col2:
                st.write(f"**Priority:** {complaint['priority']}")
            with col3:
                st.write(f"**Category:** {complaint['category_name']}")
            
            st.write(f"**Description:** {complaint['description']}")
            if complaint['location']:
                st.write(f"**Location:** {complaint['location']}")
            
            st.write(f"**Submitted:** {complaint['created_at']}")
            if complaint['assigned_to_name']:
                st.write(f"**Assigned to:** {complaint['assigned_to_name']}")
            
            # Show updates
            updates = complaint_manager.get_complaint_updates(complaint['id'])
            if updates:
                st.write("**Updates:**")
                for update in updates:
                    icon = "👤" if update['user_type'] == 'citizen' else "🏛️"
                    st.write(f"• {icon} **{update['username']}** ({update['created_at']}): {update['update_text']}")
            
            # Add comment section
            with st.form(f"comment_form_{complaint['id']}"):
                new_comment = st.text_area("Add a comment:", key=f"comment_{complaint['id']}")
                if st.form_submit_button("Add Comment"):
                    if new_comment:
                        if complaint_manager.add_complaint_comment(complaint['id'], user_id, new_comment):
                            st.success("Comment added successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to add comment")


def render_government_complaints_view(complaint_manager: ComplaintManager, user_id: str, user_type: str):
    """Render government official's complaint management view"""
    st.subheader("🏛️ Complaint Management")
    
    tab1, tab2, tab3 = st.tabs(["All Complaints", "My Assignments", "Statistics"])
    
    with tab1:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Status", ["All", "Open", "In Progress", "Resolved", "Closed"])
        with col2:
            # Get government officials for assignment filter
            c = st.session_state.db_connection.cursor()
            c.execute('''
                SELECT u.id, u.username 
                FROM users u 
                JOIN government_officials g ON u.id = g.user_id 
                WHERE g.approved = 1
            ''')
            officials = c.fetchall()
            official_options = {"All": None} | {username: uid for uid, username in officials}
            assigned_filter = st.selectbox("Assigned to", list(official_options.keys()))
        
        # Get complaints with filters
        status = status_filter if status_filter != "All" else None
        assigned_to = official_options[assigned_filter] if assigned_filter != "All" else None
        
        complaints = complaint_manager.get_all_complaints(status, assigned_to)
        
        if complaints:
            for complaint in complaints:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**#{complaint['id'][:8]}** - {complaint['title']}")
                        st.caption(f"By: {complaint['citizen_name']} | {complaint['category_name']}")
                    
                    with col2:
                        status_color = {"Open": "🔵", "In Progress": "🟡", "Resolved": "🟢", "Closed": "⚫"}
                        st.write(f"{status_color.get(complaint['status'], '⚪')} {complaint['status']}")
                    
                    with col3:
                        if complaint['assigned_to_name']:
                            st.write(f"👤 {complaint['assigned_to_name']}")
                        else:
                            st.write("Unassigned")
                    
                    with col4:
                        if st.button("Manage", key=f"manage_{complaint['id']}"):
                            st.session_state.selected_complaint = complaint['id']
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No complaints found matching the current filters.")
    
    with tab2:
        # Show complaints assigned to current user
        my_complaints = complaint_manager.get_all_complaints(assigned_to=user_id)
        
        if my_complaints:
            for complaint in my_complaints:
                with st.expander(f"#{complaint['id'][:8]} - {complaint['title']}", expanded=False):
                    st.write(f"**Citizen:** {complaint['citizen_name']}")
                    st.write(f"**Description:** {complaint['description']}")
                    st.write(f"**Location:** {complaint['location'] or 'Not specified'}")
                    st.write(f"**Priority:** {complaint['priority']}")
                    
                    # Status update form
                    with st.form(f"status_form_{complaint['id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            new_status = st.selectbox("Update Status", 
                                                    ["Open", "In Progress", "Resolved", "Closed"],
                                                    index=["Open", "In Progress", "Resolved", "Closed"].index(complaint['status']))
                        with col2:
                            comment = st.text_input("Comment (optional)")
                        
                        if st.form_submit_button("Update Status"):
                            if complaint_manager.update_complaint_status(complaint['id'], new_status, user_id, comment):
                                st.success("Status updated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to update status")
        else:
            st.info("No complaints assigned to you.")
    
    with tab3:
        # Display statistics
        stats = complaint_manager.get_complaint_statistics()
        
        st.subheader("📊 Complaint Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Complaints by Status:**")
            for status, count in stats['by_status'].items():
                st.write(f"• {status}: {count}")
        
        with col2:
            st.write("**Complaints by Category:**")
            for category, count in stats['by_category'].items():
                st.write(f"• {category}: {count}")
        
        st.metric("Average Resolution Time", f"{stats['avg_resolution_days']} days")


