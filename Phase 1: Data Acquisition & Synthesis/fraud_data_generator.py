import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import json

# Initialize Faker for realistic data generation
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

class InsuranceFraudDataGenerator:
    def __init__(self):
        self.claim_types = ['Collision', 'Property Damage', 'Bodily Injury', 'Personal Injury', 'Theft', 'Fire', 'Natural Disaster']
        self.occupations = ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Retail Worker', 'Construction Worker', 
                           'Office Manager', 'Salesperson', 'Unemployed', 'Self-employed', 'Accountant', 'Nurse']
        self.repair_shops = [f"Shop_{i}" for i in range(1, 21)]  # 20 repair shops
        self.shared_addresses = [fake.address() for _ in range(50)]  # Pool of addresses for network features
        self.shared_phones = [fake.phone_number() for _ in range(30)]  # Pool of phone numbers
        
    def generate_policy_details(self, is_fraud=False):
        """Generate policy details with potential fraud indicators"""
        # Policy tenure (fraudsters often have newer policies)
        if is_fraud and random.random() < 0.6:
            tenure_days = random.randint(1, 180)  # New policy
        else:
            tenure_days = random.randint(180, 3650)  # 6 months to 10 years
        
        # Premium calculation (fraudsters might have minimum coverage)
        base_premium = random.uniform(500, 5000)
        if is_fraud and random.random() < 0.4:
            premium = base_premium * 0.6  # Lower premium
        else:
            premium = base_premium
        
        # Coverage limits
        coverage_limit = random.choice([25000, 50000, 100000, 250000, 500000])
        if is_fraud and random.random() < 0.3:
            coverage_limit = min(coverage_limit, 50000)  # Lower coverage
        
        # Deductibles
        deductible = random.choice([250, 500, 1000, 2500, 5000])
        
        return {
            'premium': round(premium, 2),
            'tenure_days': tenure_days,
            'coverage_limit': coverage_limit,
            'deductible': deductible,
            'policy_start_date': datetime.now() - timedelta(days=tenure_days)
        }
    
    def generate_claimant_info(self, is_fraud=False):
        """Generate claimant information with potential fraud patterns"""
        age = random.randint(18, 75)
        if is_fraud and random.random() < 0.3:
            age = random.randint(22, 35)  # Fraudsters often in specific age range
        
        occupation = random.choice(self.occupations)
        
        # Location - fraudsters might cluster in certain areas
        if is_fraud and random.random() < 0.4:
            location = random.choice(self.shared_addresses[:10])  # Use shared addresses
        else:
            location = fake.address()
        
        # Claim history
        num_prior_claims = 0
        if is_fraud and random.random() < 0.5:
            num_prior_claims = random.randint(2, 8)  # More claims
        else:
            num_prior_claims = random.choices([0, 1, 2, 3], weights=[0.6, 0.25, 0.1, 0.05])[0]
        
        return {
            'age': age,
            'occupation': occupation,
            'location': location,
            'num_prior_claims': num_prior_claims,
            'customer_id': fake.uuid4()
        }
    
    def generate_claim_characteristics(self, policy_details, is_fraud=False):
        """Generate claim characteristics with fraud patterns"""
        claim_type = random.choice(self.claim_types)
        
        # Claim amount
        if is_fraud:
            if random.random() < 0.7:
                # Just under deductible or coverage limit
                if random.random() < 0.5:
                    amount = policy_details['deductible'] * random.uniform(0.85, 0.99)
                else:
                    amount = policy_details['coverage_limit'] * random.uniform(0.8, 0.95)
            else:
                amount = random.uniform(5000, 50000)
        else:
            # Normal distribution for legitimate claims
            amount = min(abs(np.random.normal(15000, 10000)), policy_details['coverage_limit'])
        
        # Claim date (relative to policy start)
        days_after_policy_start = random.randint(1, policy_details['tenure_days'])
        if is_fraud and random.random() < 0.6:
            # Fraudulent claims often happen soon after policy start
            days_after_policy_start = min(random.randint(1, 90), policy_details['tenure_days'])
        
        claim_date = policy_details['policy_start_date'] + timedelta(days=days_after_policy_start)
        
        # Description
        descriptions = {
            'Collision': ['Rear-ended at traffic light', 'Hit parked car', 'Highway collision', 'Intersection accident'],
            'Property Damage': ['Vandalism to vehicle', 'Hail damage', 'Tree fell on car', 'Shopping cart damage'],
            'Bodily Injury': ['Whiplash from collision', 'Back injury', 'Broken arm', 'Concussion'],
            'Personal Injury': ['Slip and fall', 'Dog bite', 'Assault', 'Product liability'],
            'Theft': ['Vehicle stolen from parking lot', 'Catalytic converter theft', 'Break-in theft', 'Carjacking'],
            'Fire': ['Engine fire', 'Electrical fire', 'Arson', 'Garage fire'],
            'Natural Disaster': ['Flood damage', 'Earthquake damage', 'Hurricane damage', 'Tornado damage']
        }
        
        description = random.choice(descriptions.get(claim_type, ['Generic claim']))
        
        return {
            'claim_amount': round(amount, 2),
            'claim_date': claim_date,
            'claim_type': claim_type,
            'claim_description': description,
            'claim_id': fake.uuid4()
        }
    
    def generate_behavioral_flags(self, claim_date, is_fraud=False):
        """Generate behavioral indicators"""
        # Time to report (fraudsters might report very quickly or with suspicious delays)
        if is_fraud:
            if random.random() < 0.5:
                days_to_report = random.choices([0, 1, 30, 45], weights=[0.4, 0.3, 0.2, 0.1])[0]
            else:
                days_to_report = random.randint(0, 60)
        else:
            days_to_report = random.choices([0, 1, 2, 3, 7, 14], weights=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05])[0]
        
        report_date = claim_date + timedelta(days=days_to_report)
        
        # Documentation quality (1-10 scale)
        if is_fraud:
            doc_quality = random.choices([2, 3, 4, 8, 9], weights=[0.2, 0.3, 0.3, 0.1, 0.1])[0]
        else:
            doc_quality = random.choices([6, 7, 8, 9, 10], weights=[0.1, 0.2, 0.3, 0.3, 0.1])[0]
        
        # Witness availability
        if is_fraud:
            witness_available = random.choices([True, False], weights=[0.3, 0.7])[0]
        else:
            witness_available = random.choices([True, False], weights=[0.6, 0.4])[0]
        
        # Police report filed
        police_report = random.choices([True, False], weights=[0.7, 0.3])[0]
        if is_fraud and random.random() < 0.4:
            police_report = False
        
        return {
            'days_to_report': days_to_report,
            'report_date': report_date,
            'documentation_quality': doc_quality,
            'witness_available': witness_available,
            'police_report_filed': police_report
        }
    
    def generate_network_features(self, claimant_info, is_fraud=False):
        """Generate network-related features for fraud rings"""
        features = {
            'shared_address': False,
            'shared_phone': False,
            'suspicious_repair_shop': False,
            'linked_claims': 0
        }
        
        if is_fraud:
            # Shared address (already handled in claimant_info)
            if claimant_info['location'] in self.shared_addresses[:10]:
                features['shared_address'] = True
            
            # Shared phone
            if random.random() < 0.4:
                features['shared_phone'] = True
                features['phone_number'] = random.choice(self.shared_phones[:10])
            else:
                features['phone_number'] = fake.phone_number()
            
            # Suspicious repair shop (some shops might be part of fraud rings)
            if random.random() < 0.6:
                features['repair_shop'] = random.choice(self.repair_shops[:5])  # First 5 shops are "suspicious"
                features['suspicious_repair_shop'] = True
            else:
                features['repair_shop'] = random.choice(self.repair_shops)
            
            # Linked claims (number of claims linked to same address/phone/shop)
            features['linked_claims'] = random.choices([1, 2, 3, 5, 8], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0]
        else:
            features['phone_number'] = fake.phone_number()
            features['repair_shop'] = random.choice(self.repair_shops)
            features['linked_claims'] = random.choices([0, 1], weights=[0.9, 0.1])[0]
        
        return features
    
    def generate_single_record(self, is_fraud=False):
        """Generate a complete insurance claim record"""
        # Generate all components
        policy = self.generate_policy_details(is_fraud)
        claimant = self.generate_claimant_info(is_fraud)
        claim = self.generate_claim_characteristics(policy, is_fraud)
        behavioral = self.generate_behavioral_flags(claim['claim_date'], is_fraud)
        network = self.generate_network_features(claimant, is_fraud)
        
        # Combine all features
        record = {
            # Claim characteristics
            'claim_id': claim['claim_id'],
            'claim_amount': claim['claim_amount'],
            'claim_date': claim['claim_date'].strftime('%Y-%m-%d'),
            'claim_type': claim['claim_type'],
            'claim_description': claim['claim_description'],
            
            # Policy details
            'policy_premium': policy['premium'],
            'policy_tenure_days': policy['tenure_days'],
            'coverage_limit': policy['coverage_limit'],
            'deductible': policy['deductible'],
            'policy_start_date': policy['policy_start_date'].strftime('%Y-%m-%d'),
            
            # Claimant info
            'customer_id': claimant['customer_id'],
            'age': claimant['age'],
            'occupation': claimant['occupation'],
            'location': claimant['location'],
            'num_prior_claims': claimant['num_prior_claims'],
            
            # Behavioral flags
            'days_to_report': behavioral['days_to_report'],
            'report_date': behavioral['report_date'].strftime('%Y-%m-%d'),
            'documentation_quality': behavioral['documentation_quality'],
            'witness_available': behavioral['witness_available'],
            'police_report_filed': behavioral['police_report_filed'],
            
            # Network features
            'phone_number': network['phone_number'],
            'repair_shop': network['repair_shop'],
            'shared_address': network['shared_address'],
            'shared_phone': network['shared_phone'],
            'suspicious_repair_shop': network['suspicious_repair_shop'],
            'linked_claims': network['linked_claims'],
            
            # Target variable
            'is_fraud': is_fraud
        }
        
        return record
    
    def generate_dataset(self, num_records=10000, fraud_rate=0.15):
        """Generate complete synthetic dataset"""
        records = []
        num_fraud = int(num_records * fraud_rate)
        num_legitimate = num_records - num_fraud
        
        print(f"Generating {num_records} records: {num_legitimate} legitimate, {num_fraud} fraudulent...")
        
        # Generate fraudulent records
        for _ in range(num_fraud):
            records.append(self.generate_single_record(is_fraud=True))
        
        # Generate legitimate records
        for _ in range(num_legitimate):
            records.append(self.generate_single_record(is_fraud=False))
        
        # Shuffle records
        random.shuffle(records)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Add some derived features
        df['claim_to_premium_ratio'] = df['claim_amount'] / df['policy_premium']
        df['claim_to_coverage_ratio'] = df['claim_amount'] / df['coverage_limit']
        df['early_claim'] = (df['policy_tenure_days'] < 90).astype(int)
        df['high_claim_history'] = (df['num_prior_claims'] > 2).astype(int)
        
        return df

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = InsuranceFraudDataGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(num_records=5000, fraud_rate=0.15)
    
    # Save to CSV
    df.to_csv('insurance_fraud_synthetic_data.csv', index=False)
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"\nClaim type distribution:")
    print(df['claim_type'].value_counts())
    print(f"\nAverage claim amounts:")
    print(f"Legitimate: ${df[df['is_fraud']==False]['claim_amount'].mean():,.2f}")
    print(f"Fraudulent: ${df[df['is_fraud']==True]['claim_amount'].mean():,.2f}")
    
    # Save sample data preview
    print("\nFirst 5 records:")
    print(df.head())
    
    # Save data profile
    profile = {
        'total_records': len(df),
        'fraud_rate': df['is_fraud'].mean(),
        'features': list(df.columns),
        'claim_types': df['claim_type'].unique().tolist(),
        'avg_claim_amount': df['claim_amount'].mean(),
        'avg_policy_tenure': df['policy_tenure_days'].mean()
    }
    
    with open('data_profile.json', 'w') as f:
        json.dump(profile, f, indent=2)