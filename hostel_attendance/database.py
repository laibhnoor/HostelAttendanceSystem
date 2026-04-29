from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Dict, Iterator, List

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column, sessionmaker

from config import ATTENDANCE_COOLDOWN_SECONDS, DB_PATH, DEFAULT_HOSTEL

logger = logging.getLogger(__name__)

Base = declarative_base()


def _db_url() -> str:
    """Build the SQLite database URL.

    Args:
        None.

    Returns:
        SQLite database URL.
    """
    return f"sqlite:///{DB_PATH.as_posix()}"


def _create_engine():
    """Create the SQLAlchemy engine.

    Args:
        None.

    Returns:
        SQLAlchemy engine.
    """
    return create_engine(_db_url(), connect_args={"check_same_thread": False}, future=True)


ENGINE = _create_engine()
SessionLocal = sessionmaker(bind=ENGINE, expire_on_commit=False, class_=Session)


class Student(Base):
    """Student ORM model.

    Args:
        None.

    Returns:
        None.
    """

    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    hostel: Mapped[str] = mapped_column(String, nullable=False, server_default=DEFAULT_HOSTEL)
    enrolled_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())


class Attendance(Base):
    """Attendance ORM model.

    Args:
        None.

    Returns:
        None.
    """

    __tablename__ = "attendance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String, ForeignKey("students.student_id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    confidence: Mapped[float] = mapped_column(Float, nullable=False)


@contextmanager
def _session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations.

    Args:
        None.

    Returns:
        SQLAlchemy session iterator.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Initialize the database and create tables if they do not exist.

    Args:
        None.

    Returns:
        None.
    """
    try:
        Base.metadata.create_all(ENGINE)
        with ENGINE.connect() as connection:
            result = connection.execute(text("PRAGMA table_info(students)"))
            columns = {row[1] for row in result.fetchall()}
            if "hostel" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE students ADD COLUMN hostel TEXT"
                    ),
                    {"default_hostel": DEFAULT_HOSTEL},
                )
                connection.commit()
                logger.info("Migrated students table to include hostel column")
    except SQLAlchemyError as exc:
        logger.error("Failed to initialize database: %s", exc)
        raise


def add_student(student_id: str, name: str, hostel: str) -> bool:
    """Add a student to the database.

    Args:
        student_id: Unique student identifier.
        name: Student full name.

    Returns:
        True if added, False if already exists or on error.
    """
    try:
        with _session_scope() as session:
            session.add(Student(student_id=student_id, name=name, hostel=hostel))
        return True
    except IntegrityError:
        logger.warning("Student already exists: %s", student_id)
        return False
    except SQLAlchemyError as exc:
        logger.error("Failed to add student %s: %s", student_id, exc)
        return False


def student_exists(student_id: str) -> bool:
    """Check if a student exists in the database.

    Args:
        student_id: Unique student identifier.

    Returns:
        True if student exists, False otherwise.
    """
    try:
        with _session_scope() as session:
            stmt = select(Student.student_id).where(Student.student_id == student_id)
            result = session.execute(stmt).first()
            return result is not None
    except SQLAlchemyError as exc:
        logger.error("Failed to check student %s: %s", student_id, exc)
        return False


def mark_attendance(student_id: str, conf: float) -> bool:
    """Mark attendance for a student if not recently marked.

    Args:
        student_id: Unique student identifier.
        conf: Matching confidence score.

    Returns:
        True if attendance was marked, False if skipped or on error.
    """
    try:
        with _session_scope() as session:
            stmt = (
                select(Attendance.timestamp)
                .where(Attendance.student_id == student_id)
                .order_by(Attendance.timestamp.desc())
                .limit(1)
            )
            last_record = session.execute(stmt).scalar_one_or_none()
            if last_record is not None:
                now = datetime.utcnow()
                delta = (now - last_record).total_seconds()
                if delta < ATTENDANCE_COOLDOWN_SECONDS:
                    return False

            session.add(Attendance(student_id=student_id, confidence=conf))
        return True
    except SQLAlchemyError as exc:
        logger.error("Failed to mark attendance for %s: %s", student_id, exc)
        return False


def get_attendance_today() -> List[Dict[str, str]]:
    """Get today's attendance records.

    Args:
        None.

    Returns:
        List of attendance dictionaries.
    """
    try:
        with _session_scope() as session:
            today_str = date.today().isoformat()
            stmt = (
                select(
                    Attendance.student_id,
                    Student.name,
                    Student.hostel,
                    Attendance.timestamp,
                    Attendance.confidence,
                )
                .join(Student, Student.student_id == Attendance.student_id)
                .where(func.date(Attendance.timestamp) == today_str)
                .order_by(Attendance.timestamp.desc())
            )
            rows = session.execute(stmt).all()
            return [
                {
                    "student_id": row.student_id,
                    "name": row.name,
                    "timestamp": row.timestamp,
                    "confidence": f"{row.confidence:.4f}",
                    "hostel": row.hostel,
                }
                for row in rows
            ]
    except SQLAlchemyError as exc:
        logger.error("Failed to fetch today's attendance: %s", exc)
        return []


def get_all_students() -> List[Dict[str, str]]:
    """Get all enrolled students.

    Args:
        None.

    Returns:
        List of student dictionaries.
    """
    try:
        with _session_scope() as session:
            stmt = select(Student.student_id, Student.name, Student.hostel).order_by(
                Student.student_id
            )
            rows = session.execute(stmt).all()
            return [
                {"student_id": row.student_id, "name": row.name, "hostel": row.hostel}
                for row in rows
            ]
    except SQLAlchemyError as exc:
        logger.error("Failed to fetch students: %s", exc)
        return []


def get_student_map() -> Dict[str, str]:
    """Get a mapping of student_id to name.

    Args:
        None.

    Returns:
        Dictionary mapping student_id to name.
    """
    students = get_all_students()
    return {student["student_id"]: student["name"] for student in students}


def get_student_record(student_id: str) -> Dict[str, str] | None:
    """Get a student record by student_id.

    Args:
        student_id: Unique student identifier.

    Returns:
        Student record dictionary or None if not found.
    """
    try:
        with _session_scope() as session:
            stmt = (
                select(Student.student_id, Student.name, Student.hostel)
                .where(Student.student_id == student_id)
                .limit(1)
            )
            row = session.execute(stmt).first()
            if row is None:
                return None
            return {"student_id": row.student_id, "name": row.name, "hostel": row.hostel}
    except SQLAlchemyError as exc:
        logger.error("Failed to fetch student %s: %s", student_id, exc)
        return None
