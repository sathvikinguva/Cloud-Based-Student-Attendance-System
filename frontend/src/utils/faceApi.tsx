interface FaceApiResponse {
  isIdentical: boolean;
  confidence: number;
  message?: string;
}

interface VerificationResult {
  success: boolean;
  isPresent: boolean;
  message: string;
  confidence?: number;
}

const API_URL = 'http://localhost:5000'; // Ensure this matches your backend URL

export async function detectFace(imageData: string): Promise<string> {
  try {
    console.log('Preparing image for face detection...');

    // Convert base64 to blob
    const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
    const byteCharacters = atob(base64Data);
    const byteArrays = [];

    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
      const slice = byteCharacters.slice(offset, offset + 512);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }

    const blob = new Blob(byteArrays, { type: 'image/jpeg' });

    // Create form data
    const formData = new FormData();
    formData.append('image', blob, 'face.jpg');

    // Call backend API
    const response = await fetch(`${API_URL}/api/detect-face`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Face detection failed: ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    if (data && data.length > 0) {
      return data[0].faceId;
    } else {
      throw new Error('No face detected in the image');
    }
  } catch (error) {
    console.error('Error in detectFace:', error);
    throw error;
  }
}

export async function verifyFace(
  faceId1: string, 
  faceId2: string, 
  studentId: string
): Promise<VerificationResult> {
  try {
    console.log('Verifying faces...', { faceId1, faceId2, studentId });
    
    const response = await fetch(`${API_URL}/api/verify-face`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        faceId1,
        faceId2,
        studentId
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Verification failed with status ${response.status}: ${errorText}`);
      return {
        success: false,
        isPresent: false,
        message: `Verification failed: ${response.statusText}`
      };
    }
    
    const data = await response.json();
    console.log("Verification response:", data);
    
    // Check if the response contains the expected fields
    if (data && data.hasOwnProperty('isIdentical')) {
      if (data.isIdentical === true) {
        // Student is present
        return {
          success: true,
          isPresent: true,
          message: data.message || "Student is present",
          confidence: data.confidence
        };
      } else {
        // Student is absent
        return {
          success: true,
          isPresent: false,
          message: data.message || "Student is absent",
          confidence: data.confidence
        };
      }
    } else {
      // Unexpected response format
      console.error("Unexpected verification response format:", data);
      return {
        success: false,
        isPresent: false,
        message: "Verification failed with unexpected response format"
      };
    }
  } catch (error) {
    console.error("Error during verification:", error);
    return {
      success: false,
      isPresent: false,
      message: `Verification error: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}

// Optional: Helper function to register a student face
export async function registerStudentFace(
  studentId: string, 
  imageData: string
): Promise<{ success: boolean; faceId?: string; message: string }> {
  try {
    console.log('Registering student face...');

    // Convert base64 to blob
    const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
    const byteCharacters = atob(base64Data);
    const byteArrays = [];

    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
      const slice = byteCharacters.slice(offset, offset + 512);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }

    const blob = new Blob(byteArrays, { type: 'image/jpeg' });

    // Create form data
    const formData = new FormData();
    formData.append('image', blob, 'face.jpg');
    formData.append('studentId', studentId);

    // Call backend API
    const response = await fetch(`${API_URL}/api/register-student`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Student registration failed: ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    return {
      success: true,
      faceId: data.faceId,
      message: data.message || "Student registered successfully"
    };
  } catch (error) {
    console.error('Error in registerStudentFace:', error);
    return {
      success: false,
      message: `Registration error: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}