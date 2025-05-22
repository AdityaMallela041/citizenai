class QueryClassifier:
    """
    Handles classification, prioritization, and routing of citizen queries
    """
    
    def __init__(self):
        # Define categories for city services
        self.categories = {
            "infrastructure": ["road", "bridge", "construction", "maintenance", "sidewalk", "pothole", "streetlight"],
            "public_services": ["water", "electricity", "garbage", "waste", "sewage", "billing", "payment"],
            "permits": ["license", "permit", "application", "business", "construction", "event", "registration"],
            "transportation": ["bus", "metro", "public transit", "schedule", "route", "fare", "parking"],
            "emergency": ["police", "fire", "ambulance", "disaster", "emergency", "urgent", "immediate"],
            "health": ["hospital", "clinic", "vaccination", "health service", "medical", "doctor"],
            "education": ["school", "education", "college", "university", "library", "course", "study"],
            "recreation": ["park", "garden", "playground", "sports", "recreation", "event", "festival"],
            "governance": ["council", "mayor", "election", "vote", "policy", "law", "regulation"]
        }
        
        # Define urgency keywords
        self.urgency_keywords = {
            "high": ["urgent", "emergency", "immediate", "critical", "unsafe", "danger", "accident", "life-threatening"],
            "medium": ["important", "soon", "needed", "problem", "issue", "broken", "not working", "deadline"],
            "low": ["information", "curious", "question", "wondering", "when", "how to", "where", "learn about"]
        }
    
    def classify_query(self, query_text):
        """
        Classify a query based on its content and determine its priority
        
        Args:
            query_text: The citizen's query text
            
        Returns:
            Dict with category, priority, and routing information
        """
        query_lower = query_text.lower()
        
        # Determine category
        matched_category = "general"
        highest_match_count = 0
        
        for category, keywords in self.categories.items():
            match_count = sum(1 for keyword in keywords if keyword in query_lower)
            if match_count > highest_match_count:
                highest_match_count = match_count
                matched_category = category
        
        # Determine priority
        priority = "low"  # Default priority
        
        # Check for high priority first
        for keyword in self.urgency_keywords["high"]:
            if keyword in query_lower:
                priority = "high"
                break
                
        # If not high, check for medium
        if priority == "low":
            for keyword in self.urgency_keywords["medium"]:
                if keyword in query_lower:
                    priority = "medium"
                    break
        
        # Determine routing
        department = self._get_department(matched_category)
        needs_human = self._needs_human_intervention(query_lower, priority)
        
        return {
            "category": matched_category,
            "priority": priority,
            "department": department,
            "needs_human": needs_human
        }
    
    def _get_department(self, category):
        """
        Map category to responsible department
        
        Args:
            category: Query category
            
        Returns:
            Department name
        """
        department_mapping = {
            "infrastructure": "Public Works Department",
            "public_services": "Utilities Department",
            "permits": "Permits and Licensing Department",
            "transportation": "Transit Authority",
            "emergency": "Emergency Services",
            "health": "Health Department",
            "education": "Education Department",
            "recreation": "Parks and Recreation",
            "governance": "City Manager's Office",
            "general": "Citizen Information Center"
        }
        
        return department_mapping.get(category, "Citizen Information Center")
    
    def _needs_human_intervention(self, query_text, priority):
        """
        Determine if a query needs human intervention
        
        Args:
            query_text: The query text
            priority: Query priority level
            
        Returns:
            Boolean indicating if human intervention is needed
        """
        # High priority queries typically need human intervention
        if priority == "high":
            return True
        
        # Complex queries that might need human expertise
        complex_indicators = [
            "appeal", "dispute", "complaint", "lawsuit", "legal",
            "exception", "waiver", "special case", "unique situation",
            "not satisfied", "unhappy", "angry", "frustrated"
        ]
        
        for indicator in complex_indicators:
            if indicator in query_text:
                return True
        
        # Default to automation
        return False
    
    def get_workflow_actions(self, query_classification):
        """
        Determine workflow actions based on query classification
        
        Args:
            query_classification: Dict with query classification details
            
        Returns:
            Dict with workflow actions
        """
        actions = {}
        
        # Set response time based on priority
        if query_classification["priority"] == "high":
            actions["response_time"] = "immediate"
            actions["escalation"] = True
        elif query_classification["priority"] == "medium":
            actions["response_time"] = "same day"
            actions["escalation"] = False
        else:
            actions["response_time"] = "standard"
            actions["escalation"] = False
        
        # Set routing action
        if query_classification["needs_human"]:
            actions["routing"] = f"Forward to {query_classification['department']} staff"
            actions["notification"] = "Email and SMS"
        else:
            actions["routing"] = "Automated response"
            actions["notification"] = "Email only"
        
        # Set follow-up action
        if query_classification["priority"] == "high":
            actions["follow_up"] = "24 hours"
        elif query_classification["category"] in ["permits", "public_services"]:
            actions["follow_up"] = "3 days"
        else:
            actions["follow_up"] = "none"
            
        return actions