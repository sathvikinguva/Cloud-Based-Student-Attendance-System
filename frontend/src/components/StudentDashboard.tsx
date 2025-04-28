import React, { useState, useEffect } from 'react';
import { Student, AttendanceRecord } from '../types';
import Camera from './Camera';
import { LogOut, UserCircle, AlertCircle } from 'lucide-react';
import { detectFace, verifyFace } from '../utils/faceApi';

interface StudentDashboardProps {
  student: Student;
  onLogout: () => void;
}

export default function StudentDashboard({ student, onLogout }: StudentDashboardProps) {
  const [isAttendanceActive, setIsAttendanceActive] = useState(false);
  const [attendanceRecords, setAttendanceRecords] = useState<AttendanceRecord[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [statusType, setStatusType] = useState<'success' | 'error' | 'info'>('info');

  // Load attendance records for the current student
  useEffect(() => {
    const records = JSON.parse(localStorage.getItem('attendanceRecords') || '[]');
    setAttendanceRecords(records.filter((record: AttendanceRecord) => record.studentId === student.id));
  }, [student.id]);

  // Check if attendance is active
  useEffect(() => {
    const checkAttendanceStatus = () => {
      const status = localStorage.getItem('attendanceStatus');
      setIsAttendanceActive(status === 'active');
    };

    window.addEventListener('storage', checkAttendanceStatus);
    checkAttendanceStatus();

    return () => window.removeEventListener('storage', checkAttendanceStatus);
  }, []);

  const showMessage = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    setStatusMessage(message);
    setStatusType(type);
    // Clear the message after 5 seconds
    setTimeout(() => setStatusMessage(null), 5000);
  };

  // Handle image capture and process it
  const handleCapture = async (imageData: string) => {
    setIsProcessing(true);
    showMessage("Processing your image...", 'info');

    try {
      if (student.isFirstLogin) {
        console.log("First login: Detecting face to register...");
        const faceId = await detectFace(imageData);
        console.log("Detected face ID for registration:", faceId);

        const updatedStudent = {
          ...student,
          faceImage: imageData,
          faceId: faceId,
          isFirstLogin: false,
        };

        const users = JSON.parse(localStorage.getItem('users') || '[]');
        const updatedUsers = users.map((u: Student) =>
          u.id === student.id ? updatedStudent : u
        );

        localStorage.setItem('users', JSON.stringify(updatedUsers));
        console.log("Face registered successfully");
        showMessage("Face image saved successfully! You can now mark attendance.", 'success');
      } else if (isAttendanceActive) {
        console.log("Attendance marking: Detecting current face...");
        const newFaceId = await detectFace(imageData);
        console.log("Detected current face ID:", newFaceId);

        const users = JSON.parse(localStorage.getItem('users') || '[]');
        const currentStudent = users.find((u: Student) => u.id === student.id);

        if (!currentStudent || !currentStudent.faceId) {
          throw new Error("No registered face found. Please register again.");
        }

        console.log("Verifying detected face with registered face...");
        console.log("Registered face ID:", currentStudent.faceId);
        console.log("Current face ID:", newFaceId);
        
        const verificationResult = await verifyFace(currentStudent.faceId, newFaceId, student.id);
        console.log("Verification result:", verificationResult);

        if (verificationResult.success) {
          if (verificationResult.isPresent) {
            console.log("Face verification successful. Marking attendance as PRESENT");

            const newRecord: AttendanceRecord = {
              date: new Date().toLocaleDateString(),
              studentId: student.id,
              studentName: student.name,
              status: 'present',
              timestamp: new Date().toISOString(),
            };

            const records = JSON.parse(localStorage.getItem('attendanceRecords') || '[]');
            localStorage.setItem('attendanceRecords', JSON.stringify([...records, newRecord]));
            setAttendanceRecords([...attendanceRecords, newRecord]);
            showMessage("Attendance marked successfully! You are PRESENT.", 'success');
          } else {
            console.log("Face verification failed. Marking attendance as ABSENT");
            
            const newRecord: AttendanceRecord = {
              date: new Date().toLocaleDateString(),
              studentId: student.id,
              studentName: student.name,
              status: 'absent',
              timestamp: new Date().toISOString(),
            };

            const records = JSON.parse(localStorage.getItem('attendanceRecords') || '[]');
            localStorage.setItem('attendanceRecords', JSON.stringify([...records, newRecord]));
            setAttendanceRecords([...attendanceRecords, newRecord]);
            showMessage("Face verification failed. You are marked ABSENT.", 'error');
          }
        } else {
          // This is for cases where the verification API call failed
          console.error("Verification API call failed:", verificationResult.message);
          showMessage(`Verification error: ${verificationResult.message}`, 'error');
        }
      } else {
        showMessage("Attendance marking is not currently active.", 'info');
      }
    } catch (error) {
      console.error("Error in face processing:", error);
      showMessage(`Error: ${(error as Error).message}`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center">
              <UserCircle size={24} className="text-gray-500" />
              <span className="ml-2 font-semibold text-gray-900">{student.name}</span>
            </div>
            <button
              onClick={onLogout}
              className="flex items-center text-gray-500 hover:text-gray-700"
            >
              <LogOut size={20} />
              <span className="ml-2">Logout</span>
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {statusMessage && (
            <div
              className={`mb-4 p-4 rounded-md flex items-center ${
                statusType === 'error' 
                  ? 'bg-red-100 text-red-700' 
                  : statusType === 'success'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-blue-100 text-blue-700'
              }`}
            >
              {statusType === 'error' && (
                <AlertCircle size={20} className="mr-2 flex-shrink-0" />
              )}
              <span>{statusMessage}</span>
            </div>
          )}

          {student.isFirstLogin ? (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Welcome! Please capture your face image</h2>
              <p className="mb-4 text-gray-600">
                This will be used for attendance verification. Please make sure your face is clearly visible,
                well-lit, and look directly at the camera.
              </p>
              <Camera onCapture={handleCapture} />
              {isProcessing && <p className="mt-2 text-gray-600">Processing...</p>}
            </div>
          ) : (
            <div className="space-y-6">
              {isAttendanceActive ? (
                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-xl font-semibold mb-4">Mark Attendance</h2>
                  <p className="mb-4 text-gray-600">
                    Attendance is currently active. Please look at the camera to verify your presence.
                  </p>
                  <Camera onCapture={handleCapture} />
                  {isProcessing && <p className="mt-2 text-gray-600">Verifying your face...</p>}
                </div>
              ) : (
                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <AlertCircle className="h-5 w-5 text-yellow-400" aria-hidden="true" />
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-yellow-700">
                        Attendance marking is not active at this time. Please try again later.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <div className="bg-white rounded-lg shadow overflow-hidden">
                <div className="px-4 py-5 sm:p-6">
                  <h3 className="text-lg font-medium text-gray-900">Attendance History</h3>
                  {attendanceRecords.length > 0 ? (
                    <div className="mt-4">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Date
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Status
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Time
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {attendanceRecords.map((record, index) => (
                            <tr key={index}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                {record.date}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm">
                                <span 
                                  className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                                  ${record.status === 'present' 
                                    ? 'bg-green-100 text-green-800' 
                                    : 'bg-red-100 text-red-800'
                                  }`}
                                >
                                  {record.status.toUpperCase()}
                                </span>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                {new Date(record.timestamp).toLocaleTimeString()}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="mt-4 text-gray-500">No attendance records found.</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}