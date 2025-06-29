import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

# Initialize with seeds for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

class ImprovedInsuranceFraudDataGenerator:
    """
    Improved generator with more realistic fraud patterns
    Key changes:
    - No perfect correlations
    - More subtle fraud indicators
    - Realistic overlap between fraud and legitimate claims
    - Added noise and variability
    """
    
    def __init__(self):
        self.claim_types = ['Collision', 'Property Damage', 'Bodily Injury', 'Personal Injury', 'Theft', 'Fire', 'Natural Disaster']
        self.occupations = ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Retail Worker', 'Construction Worker', 
                           'Office Manager', 'Salesperson', 'Unemployed', 'Self-employed', 'Accountant', 'Nurse']
        self.repair_shops = [f"Shop_{i}" for i in range(1, 21)]
        self.shared_addresses = [fake.address() for _ in range(50)]
        self.shared_phones = [fake.phone_number() for _ in range(30)]
        
    def generate_policy_details(self, is_fraud=False):
        """Generate policy details with more realistic patterns"""
        # Policy tenure - fraudsters TEND to have newer policies but not always
        if is_fraud and random.random() < 0.35:  # Only 35% of fraudsters have very new policies
            tenure_days = random.randint(1, 180)
        else:
            tenure_days = random.randint(30, 3650)  # More overlap
        
        # Premium calculation - more subtle differences
        base_premium = random.uniform(500, 5000)
        if is_fraud and random.random() < 0.25:  # Only 25% have suspiciously low premiums
            premium = base_premium * 0.8
        else:
            premium = base_premium * random.uniform(0.9, 1.1)
        
        # Coverage limits - more variation
        coverage_limit = random.choice([25000, 50000, 100000, 250000, 500000])
        if is_fraud and random.random() < 0.2:  # Only 20% target specific limits
            coverage_limit = min(coverage_limit, 100000)
        
        deductible = random.choice([250, 500, 1000, 2500, 5000])
        
        return {
            'premium': round(premium, 2),
            'tenure_days': tenure_days,
            'coverage_limit': coverage_limit,
            'deductible': deductible,
            'policy_start_date': datetime.now() - timedelta(days=tenure_days)
        }
    
    def generate_claimant_info(self, is_fraud=False):
        """Generate claimant info with realistic patterns"""
        # Age - more overlap between fraud and legitimate
        if is_fraud and random.random() < 0.2:
            age = random.randint(22, 35)
        else:
            age = random.randint(18, 75)
        
        occupation = random.choice(self.occupations)
        
        # Location - only some fraudsters use shared addresses
        if is_fraud and random.random() < 0.15:  # Only 15% use shared addresses
            location = random.choice(self.shared_addresses[:10])
        else:
            location = fake.address()
        
        # Claim history - more realistic distribution
        if is_fraud and random.random() < 0.3:
            num_prior_claims = random.choices([2, 3, 4, 5], weights=[0.4, 0.3, 0.2, 0.1])[0]
        else:
            num_prior_claims = random.choices([0, 1, 2, 3], weights=[0.5, 0.3, 0.15, 0.05])[0]
        
        return {
            'age': age,
            'occupation': occupation,
            'location': location,
            'num_prior_claims': num_prior_claims,
            'customer_id': fake.uuid4()
        }
    
    def generate_claim_characteristics(self, policy_details, is_fraud=False):
        """Generate claim characteristics with realistic patterns"""
        claim_type = random.choice(self.claim_types)
        
        # Claim amount - more realistic distribution with overlap
        if is_fraud:
            # Mix of different fraud strategies
            strategy = random.choices(['just_under', 'high_value', 'normal_looking'], 
                                    weights=[0.3, 0.3, 0.4])[0]
            
            if strategy == 'just_under':
                # Just under limits but with noise
                if random.random() < 0.5:
                    amount = policy_details['deductible'] * random.uniform(0.85, 1.15)
                else:
                    amount = policy_details['coverage_limit'] * random.uniform(0.7, 0.95)
            elif strategy == 'high_value':
                # High value claims but not always maximum
                amount = random.uniform(20000, min(80000, policy_details['coverage_limit']))
            else:
                # Normal looking fraudulent claims
                amount = abs(np.random.normal(18000, 8000))
        else:
            # Legitimate claims - wider distribution
            amount = min(abs(np.random.normal(15000, 9000)), policy_details['coverage_limit'])
            # Some legitimate claims can be high too
            if random.random() < 0.05:  # 5% of legitimate claims are high value
                amount = random.uniform(30000, min(60000, policy_details['coverage_limit']))
        
        # Claim date - more realistic timing
        days_after_policy_start = random.randint(1, policy_details['tenure_days'])
        if is_fraud and random.random() < 0.25:  # Only 25% of frauds are suspiciously early
            days_after_policy_start = min(random.randint(1, 120), policy_details['tenure_days'])
        
        claim_date = policy_details['policy_start_date'] + timedelta(days=days_after_policy_start)
        
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
        """Generate behavioral indicators with realistic variation"""
        # Time to report - more variation
        if is_fraud:
            # Mix of behaviors
            if random.random() < 0.3:
                days_to_report = random.choices([0, 1, 20, 30, 45], weights=[0.3, 0.2, 0.2, 0.2, 0.1])[0]
            else:
                days_to_report = random.randint(0, 14)  # Some fraudsters report normally
        else:
            days_to_report = random.choices([0, 1, 2, 3, 7, 14, 30], weights=[0.25, 0.25, 0.2, 0.15, 0.1, 0.04, 0.01])[0]
        
        report_date = claim_date + timedelta(days=days_to_report)
        
        # Documentation quality - more overlap
        if is_fraud:
            # Not all fraudsters have poor documentation
            doc_quality = random.choices([2, 3, 4, 5, 6, 7, 8], 
                                       weights=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1])[0]
        else:
            # Some legitimate claims have poor documentation too
            doc_quality = random.choices([4, 5, 6, 7, 8, 9, 10], 
                                       weights=[0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.05])[0]
        
        # Witness availability - more realistic
        if is_fraud:
            witness_available = random.choices([True, False], weights=[0.25, 0.75])[0]
        else:
            witness_available = random.choices([True, False], weights=[0.55, 0.45])[0]
        
        # Police report - both groups file reports
        if is_fraud:
            police_report = random.choices([True, False], weights=[0.4, 0.6])[0]
        else:
            police_report = random.choices([True, False], weights=[0.7, 0.3])[0]
        
        return {
            'days_to_report': days_to_report,
            'report_date': report_date,
            'documentation_quality': doc_quality,
            'witness_available': witness_available,
            'police_report_filed': police_report
        }
    
    def generate_network_features(self, claimant_info, is_fraud=False):
        """Generate network features with realistic patterns"""
        features = {
            'shared_address': False,
            'shared_phone': False,
            'suspicious_repair_shop': False,
            'linked_claims': 0
        }
        
        if is_fraud:
            # Only some fraudsters are part of networks
            if random.random() < 0.15:  # 15% are part of address networks
                features['shared_address'] = True
            
            # Shared phone - even rarer
            if random.random() < 0.1:  # 10% share phones
                features['shared_phone'] = True
                features['phone_number'] = random.choice(self.shared_phones[:10])
            else:
                features['phone_number'] = fake.phone_number()
            
            # Suspicious repair shop - some correlation but not perfect
            if random.random() < 0.35:  # 35% use suspicious shops
                features['repair_shop'] = random.choice(self.repair_shops[:5])
                features['suspicious_repair_shop'] = True
            else:
                features['repair_shop'] = random.choice(self.repair_shops)
            
            # Linked claims - more realistic distribution
            if features['shared_address'] or features['shared_phone']:
                features['linked_claims'] = random.choices([1, 2, 3, 4], weights=[0.4, 0.3, 0.2, 0.1])[0]
            else:
                features['linked_claims'] = random.choices([0, 1], weights=[0.7, 0.3])[0]
        else:
            # Legitimate claims can sometimes have these features too
            features['phone_number'] = fake.phone_number()
            features['repair_shop'] = random.choice(self.repair_shops)
            
            # Very rarely, legitimate claims might appear linked
            if random.random() < 0.02:  # 2% false positive rate
                features['linked_claims'] = 1
            
            # Legitimate customers might use popular repair shops
            if features['repair_shop'] in self.repair_shops[:5] and random.random() < 0.1:
                features['suspicious_repair_shop'] = True
        
        return features
    
    def generate_single_record(self, is_fraud=False):
        """Generate a complete insurance claim record"""
        # Generate all components
        policy = self.generate_policy_details(is_fraud)
        claimant = self.generate_claimant_info(is_fraud)
        claim = self.generate_claim_characteristics(policy, is_fraud)
        behavioral = self.generate_behavioral_flags(claim['claim_date'], is_fraud)
        network = self.generate_network_features(claimant, is_fraud)
        
        # Add some noise - occasionally mislabel (simulates real-world uncertainty)
        if random.random() < 0.02:  # 2% noise rate
            is_fraud = not is_fraud
        
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
        """Generate complete synthetic dataset with realistic patterns"""
        records = []
        num_fraud = int(num_records * fraud_rate)
        num_legitimate = num_records - num_fraud
        
        print(f"Generating {num_records} records: {num_legitimate} legitimate, {num_fraud} fraudulent...")
        print("Creating more realistic fraud patterns with:")
        print("- No perfect correlations")
        print("- Overlapping distributions")
        print("- Natural variation and noise")
        print("- Subtle fraud indicators")
        
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
        
        # Add derived features
        df['claim_to_premium_ratio'] = df['claim_amount'] / df['policy_premium']
        df['claim_to_coverage_ratio'] = df['claim_amount'] / df['coverage_limit']
        df['early_claim'] = (df['policy_tenure_days'] < 90).astype(int)
        df['high_claim_history'] = (df['num_prior_claims'] > 2).astype(int)
        
        # Add some additional realistic patterns
        # Seasonal adjustments (more claims in certain months)
        df['claim_date'] = pd.to_datetime(df['claim_date'])
        month_weights = [0.8, 0.85, 1.1, 1.15, 1.2, 1.3, 0.9, 0.85, 0.9, 0.95, 0.85, 0.9]
        df['seasonal_factor'] = df['claim_date'].dt.month.map(dict(enumerate(month_weights, 1)))
        
        return df

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = ImprovedInsuranceFraudDataGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(num_records=5000, fraud_rate=0.15)
    
    # Save to CSV
    df.to_csv('insurance_fraud_realistic_data.csv', index=False)
    
    # Display statistics showing more realistic patterns
    print("\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    
    print("\nFraud rates by key indicators (showing realistic overlap):")
    print(f"Early claims (<90 days): {df[df['early_claim']==1]['is_fraud'].mean():.1%}")
    print(f"Shared addresses: {df[df['shared_address']==True]['is_fraud'].mean():.1%}")
    print(f"Low documentation (<5): {df[df['documentation_quality']<5]['is_fraud'].mean():.1%}")
    print(f"Suspicious repair shops: {df[df['suspicious_repair_shop']==True]['is_fraud'].mean():.1%}")
    
    print("\nAverage claim amounts:")
    print(f"Legitimate: ${df[df['is_fraud']==False]['claim_amount'].mean():,.2f}")
    print(f"Fraudulent: ${df[df['is_fraud']==True]['claim_amount'].mean():,.2f}")
    print(f"Ratio: {df[df['is_fraud']==True]['claim_amount'].mean() / df[df['is_fraud']==False]['claim_amount'].mean():.2f}x")
    
    print("\nFirst 5 records:")
    print(df.head())
