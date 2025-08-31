# 📊 Data Lineage Manifest

## **Purpose**
This manifest demonstrates RLDK's data lineage tracking capabilities, showing how data flows through the training pipeline and preventing contamination issues.

## **Data Flow Overview**
```
Raw Data → Preprocessing → Training Split → Training → Validation
    ↓           ↓            ↓           ↓         ↓
  Hash A    Hash B      Hash C      Hash D    Hash E
```

## **Content-Addressed Data**

### **1. Raw Data (Hash: a1b2c3d4...)**
- **Source**: Human preference datasets
- **Size**: 10,000 examples
- **Format**: JSON with human ratings
- **Checksum**: `sha256:a1b2c3d4e5f6...`

### **2. Preprocessed Data (Hash: b2c3d4e5...)**
- **Input**: Raw data (Hash: a1b2c3d4...)
- **Transformations**: Tokenization, cleaning, formatting
- **Size**: 9,800 examples (200 filtered out)
- **Checksum**: `sha256:b2c3d4e5f6g7...`

### **3. Training Split (Hash: c3d4e5f6...)**
- **Input**: Preprocessed data (Hash: b2c3d4e5...)
- **Split**: 80% training, 20% validation
- **Training**: 7,840 examples
- **Validation**: 1,960 examples
- **Checksum**: `sha256:c3d4e5f6g7h8...`

### **4. Training Data (Hash: d4e5f6g7...)**
- **Input**: Training split (Hash: c3d4e5f6...)
- **Augmentation**: Random sampling, noise injection
- **Size**: 7,840 examples
- **Checksum**: `sha256:d4e5f6g7h8i9...`

### **5. Validation Data (Hash: e5f6g7h8...)**
- **Input**: Validation split (Hash: c3d4e5f6...)
- **Augmentation**: None (kept clean)
- **Size**: 1,960 examples
- **Checksum**: `sha256:e5f6g7h8i9j0...`

## **Intentional Contamination for RLDK to Detect**

### **Bug: Data Leakage in Summarization Task**
- **What happens**: 100 validation examples accidentally included in training
- **Expected RLDK detection**: 0.92 correlation between train/val metrics
- **Fix**: Proper data splitting with content addressing

### **Bug: Non-Deterministic Preprocessing**
- **What happens**: Random seed not set in data augmentation
- **Expected RLDK detection**: Different hashes for same input data
- **Fix**: Set deterministic seeds in all data operations

### **Bug: Dataset Version Mismatch**
- **What happens**: Training uses old version, validation uses new version
- **Expected RLDK detection**: Hash mismatch between train/val sources
- **Fix**: Use same dataset version for both splits

## **RLDK Data Lineage Commands**

### **Check Data Integrity**
```bash
# Verify data hasn't been corrupted
rldk data-integrity --manifest data_lineage.md --data-dir datasets/

# Check for contamination
rldk data-contamination --train train_data.jsonl --val val_data.jsonl

# Validate lineage
rldk validate-lineage --manifest data_lineage.md --pipeline training_pipeline.yaml
```

### **Generate New Lineage**
```bash
# Create new data lineage after preprocessing
rldk create-lineage --input raw_data.jsonl --output processed_data.jsonl --transformations tokenize,clean,split

# Update manifest with new hashes
rldk update-manifest --manifest data_lineage.md --new-hash abc123...
```

## **Expected RLDK Outputs**

### **1. Data Integrity Report**
- File: `data_integrity_report.md`
- Shows: All data hashes match expected values
- Status: ✅ Passed

### **2. Contamination Detection**
- File: `contamination_report.md`
- Shows: 0.92 train/val correlation (contamination detected)
- Status: 🚨 Failed

### **3. Lineage Validation**
- File: `lineage_validation.md`
- Shows: Hash mismatches in training pipeline
- Status: 🚨 Failed

## **Benefits of Data Lineage**

### **Immediate Value**
- ✅ **Prevents contamination**: Catch data leaks before training
- ✅ **Ensures reproducibility**: Same data always produces same hashes
- ✅ **Tracks changes**: See exactly what changed in data pipeline

### **Long-term Value**
- ✅ **Audit trail**: Complete history of data transformations
- ✅ **Debugging**: Quickly identify where data issues originate
- ✅ **Compliance**: Meet data governance requirements

## **Integration with RLDK**

### **Automatic Detection**
RLDK automatically:
- Tracks data hashes throughout training
- Detects when validation data appears in training
- Warns about non-deterministic data operations
- Generates lineage reports for debugging

### **Manual Validation**
Researchers can:
- Check data integrity before training
- Validate lineage after preprocessing
- Debug contamination issues with lineage reports
- Ensure reproducible data pipelines

## **Success Criteria**
- RLDK detects all 3 data lineage bugs
- Generates actionable contamination reports
- Provides clear fix recommendations
- Demonstrates value of data lineage tracking