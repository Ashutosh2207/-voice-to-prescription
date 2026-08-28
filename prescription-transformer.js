function transformPrescription(input) {
  const frequencyMap = {
    '1-0-1': '1-0-1-0 / MORNING/AFTERNOON/EVENING/NIGHT',
    '1-1-1': '1-1-1-0 / MORNING/AFTERNOON/EVENING/NIGHT',
    '1-0-0': '1-0-0-0 / MORNING/AFTERNOON/EVENING/NIGHT',
    '0-1-0': '0-1-0-0 / MORNING/AFTERNOON/EVENING/NIGHT',
    '0-0-1': '0-0-1-0 / MORNING/AFTERNOON/EVENING/NIGHT'
  };

  const rawTranscript = input.prescription.rawTranscript || '';

  function detectForm(medicineName, transcript) {
    const nameLower = medicineName.toLowerCase();
    const transcriptLower = transcript.toLowerCase();
    
    // Check for form keywords near the medicine name in transcript
    const medicineIndex = transcriptLower.indexOf(nameLower);
    const contextStart = Math.max(0, medicineIndex - 50);
    const contextEnd = Math.min(transcriptLower.length, medicineIndex + nameLower.length + 50);
    const context = transcriptLower.substring(contextStart, contextEnd);
    
    if (context.includes('capsule')) return 'CAPSULE';
    if (context.includes('tablet')) return 'TABLET';
    if (context.includes('syrup')) return 'SYRUP';
    if (context.includes('injection')) return 'INJECTION';
    if (nameLower.includes('capsule')) return 'CAPSULE';
    if (nameLower.includes('tablet')) return 'TABLET';
    return 'TABLET';
  }

  return {
    prescription: input.prescription.medicines.map(med => ({
      name: med.medicineName.toUpperCase(),
      form: detectForm(med.medicineName, rawTranscript),
      dosage: med.dosage.toUpperCase().replace(' ', ''),
      numberOfTime: frequencyMap[med.frequency] || `${med.frequency}-0 / MORNING/AFTERNOON/EVENING/NIGHT`,
      remarks: med.instructions.toUpperCase(),
      timesPerDay: med.duration.toUpperCase(),
      unitPerTime: med.frequency.split('-')[0]
    }))
  };
}

// Example usage:
const input = {
  "prescription": {
    "medicines": [
      {
        "medicineName": "Aspirin",
        "dosage": "100 mg",
        "frequency": "1-0-1",
        "duration": "2 days",
        "route": "oral",
        "instructions": "after food",
        "confidence": 1.0
      },
      {
        "medicineName": "Paracetamol",
        "dosage": "500 mg",
        "frequency": "1-1-1",
        "duration": "3 days",
        "route": "oral",
        "instructions": "after food",
        "confidence": 1.0
      },
      {
        "medicineName": "Amoxicillin",
        "dosage": "500 mg",
        "frequency": "1-0-1",
        "duration": "5 days",
        "route": "oral",
        "instructions": "after food",
        "confidence": 1.0
      }
    ],
    "notes": null,
    "rawTranscript": "PRESCRIBE ASPIRIN 100 mg TABLET ,ONE TABLET IN THE MORNING AND ONE IN THE EVENING,AFTER FOOD FOR 2 DAYS.PARACETAMOL 500mg TABLET,ONE TABLET IN THE MRONING,AFTERNOON AND EVENING,after food,for 3 days,AMOXICILLIN 500mg CAPSULE,ONE CAPSULE INTHE MORNING AND ONE IN THE EVENING,AFTER FOOD FOR 5 DAYS"
  },
  "validation": {
    "isValid": true,
    "issues": [],
    "missingFields": []
  },
  "processingTimeMs": 38480.4208278656,
  "sttConfidence": null
};

const output = transformPrescription(input);
console.log(JSON.stringify(output, null, 2));