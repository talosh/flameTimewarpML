require "matrix"

module FlameChannelParser::Segments #:nodoc:
  
  # This segment just stays on the value of it's keyframe
  class ConstantSegment #:nodoc:
  
    NEG_INF = (-1.0/0.0)
    POS_INF = (1.0/0.0)
    
    attr_reader :start_frame, :end_frame
    
    # Tells whether this segment defines the value of the function at this time T
    def defines?(frame)
      (frame < end_frame) && (frame >= start_frame)
    end
    
    # Returns the value at this time T
    def value_at(frame)
      @v1
    end
    
    def initialize(from_frame, to_frame, value)
      @start_frame = from_frame
      @end_frame = to_frame
      @v1 = value
    end
  end
  
  # This segment linearly interpolates
  class LinearSegment < ConstantSegment #:nodoc:
  
    def initialize(from_frame, to_frame, value1, value2)
      @vint = (value2 - value1)
      super(from_frame, to_frame, value1)
    end
  
    # Returns the value at this time T
    def value_at(frame)
      on_t_interval = (frame - @start_frame).to_f / (@end_frame - @start_frame)
      @v1 + (on_t_interval * @vint)
    end
  end
  
  # This segment does Hermite interpolation
  # using the Flame algo.
  class HermiteSegment < LinearSegment #:nodoc:
  
    # In Ruby matrix columns are arrays, so here we go
    HERMATRIX = Matrix[
      [2,  -3,   0,  1],
      [-2,  3,   0,  0],
      [1,   -2,  1,  0],
      [1,   -1,  0,  0]
    ].transpose
  
    def initialize(from_frame, to_frame, value1, value2, tangent1, tangent2)
      
      @start_frame, @end_frame = from_frame, to_frame
      
      frame_interval = (@end_frame - @start_frame)
      
      # Default tangents in flame are 0, so when we do nil.to_f this is what we will get
      # CC = {P1, P2, T1, T2}
      p1, p2, t1, t2 = value1, value2, tangent1.to_f * frame_interval, tangent2.to_f * frame_interval
      @hermite = Vector[p1, p2, t1, t2]
      @basis = HERMATRIX * @hermite
    end
  
    # P[s_] = S[s].h.CC where s is 0..1 float interpolant on T (interval)
    def value_at(frame)
      return @hermite[0] if frame == @start_frame
      
      # Get the 0 < T < 1 interval we will interpolate on
      # Q[frame_] = P[ ( frame - 149 ) / (time_to - time_from)]
      t = (frame - @start_frame).to_f / (@end_frame - @start_frame)
    
      # S[s_] = {s^3, s^2, s^1, s^0}
      multipliers_vec = Vector[t ** 3,  t ** 2, t ** 1, t ** 0]
    
      # P[s_] = S[s].h.CC --> Kaboom!
      interpolated_scalar = dot_product(@basis, multipliers_vec)
    end
  
    private
  
    def dot_product(one, two)
      sum = 0.0
      (0...one.size).each { |i|  sum += one[i] * two[i] }
      sum
    end
  
  end
  
  class BezierSegment < LinearSegment #:nodoc:
    Pt = Struct.new(:x, :y, :tanx, :tany)
    
    def initialize(x1, x2, y1, y2, t1x, t1y, t2x, t2y)
      @start_frame, @end_frame = x1, x2
      
      @a = Pt.new(x1, y1, t1x, t1y)
      @b = Pt.new(x2, y2, t2x, t2y)
    end
    
    def value_at(frame)
      return @a.y if frame == @start_frame
      
      # Solve T from X. This determines the correlation between X and T.
      t = approximate_t(frame, @a.x, @a.tanx, @b.tanx, @b.x)
      vy = bezier(t, @a.y, @a.tany, @b.tany, @b.y)
    end
    
    private
    
    # t is the T interpolant (0 < T < 1)
    # a is the coordinate of the starting vertex
    # b is the coordinate of the left tangent handle
    # c is the coordinate of the right tangent handle
    # d is the coordinate of right vertex
    def bezier(t, a, b, c, d)
      a + (a*(-3) + b*3)*(t) + (a*3 - b*6 + c*3)*(t**2) + (-a + b*3 - c*3 + d)*(t**3)
    end
    
    def clamp(value)
       return 0.0 if value < 0
       return 1.0 if value > 1
       return value
    end
    
    
    APPROXIMATION_EPSILON = 1.0e-09 
    VERYSMALL = 1.0e-20 
    MAXIMUM_ITERATIONS = 100
    
    # This is how OPENCOLLADA suggests approximating Bezier animation curves
    # http://www.collada.org/public_forum/viewtopic.php?f=12&t=1132
    # Returns the approximated parameter of a parametric curve for the value X
    # @param atX At which value should the parameter be evaluated
    # @param p0x The first interpolation point of a curve segment
    # @param c0x The first control point of a curve segment
    # @param c1x The second control point of a curve segment
    # @param P1_x The second interpolation point of a curve segment
    # @return The parametric argument that is used to retrieve atX using the parametric function representation of this curve
    def approximate_t (atX, p0x, c0x, c1x, p1x )
      
      return 0.0 if (atX - p0x < VERYSMALL)
      return 1.0 if  (p1x - atX < VERYSMALL)
      
      u, v = 0.0, 1.0
      
      #  iteratively apply subdivision to approach value atX
      MAXIMUM_ITERATIONS.times do
      
         # de Casteljau Subdivision. 
         a = (p0x + c0x) / 2.0
         b = (c0x + c1x) / 2.0 
         c = (c1x + p1x) / 2.0
         d = (a + b) / 2.0 
         e = (b + c) / 2.0 
         f = (d + e) / 2.0 # this one is on the curve!
      
         # The curve point is close enough to our wanted atX
         if ((f - atX).abs < APPROXIMATION_EPSILON) 
            return clamp((u + v)*0.5)
         end
      
         # dichotomy
         if (f < atX)
            p0x = f
            c0x = e 
            c1x = c 
            u = (u + v) / 2.0 
         else
            c0x = a
            c1x = d
            p1x = f
            v = (u + v) / 2.0 
         end 
      end 
      
      clamp((u + v) / 2.0) 
    end
  end
  
  # This segment does prepolation of a constant value
  class ConstantPrepolate < LinearSegment #:nodoc:
    def initialize(upto_frame, base_value)
      @value = base_value
      @end_frame = upto_frame
      @start_frame = NEG_INF
    end
    
    def value_at(frame)
      @value
    end
  end
  
  # This segment does prepolation with a linear coefficient
  class LinearPrepolate < LinearSegment #:nodoc:
    def initialize(upto_frame, base_value, tangent)
      @value = base_value
      @end_frame = upto_frame
      @start_frame = NEG_INF
      @tangent = tangent.to_f
    end
    
    def value_at(frame)
      frame_diff = (frame - @end_frame)
      @value + (@tangent * frame_diff)
    end
  end
  
  # This segment does extrapolation using a constant value
  class ConstantExtrapolate < LinearSegment #:nodoc:
    def initialize(from_frame, base_value)
      @start_frame = from_frame
      @base_value = base_value
      @end_frame = POS_INF
    end
  
    def value_at(frame)
      @base_value
    end
  end
  
  # This segment does extrapolation using the tangent from the preceding keyframe
  class LinearExtrapolate < ConstantExtrapolate #:nodoc:
    def initialize(from_frame, base_value, tangent)
      super(from_frame, base_value)
      @tangent = tangent.to_f
    end
  
    def value_at(frame)
      frame_diff = (frame - @start_frame)
      @base_value + (@tangent * frame_diff)
    end
  end
  
  # This can be used for an anim curve that stays constant all along
  class ConstantFunction < ConstantSegment #:nodoc:
  
    def defines?(frame)
      true
    end
  
    def initialize(value)
      @value = value
    end
  
    def value_at(frame)
      @value
    end
  end
end
